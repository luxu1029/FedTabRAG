#!/usr/bin/env python3
"""FedE4RAG baseline with company clients, LoRA, teacher MSE and vanilla FedAvg."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


class Encoder(nn.Module):
    def __init__(self, model_path, rank, alpha, dropout, targets):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        replaced = []
        for name, module in list(self.backbone.named_modules()):
            if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
                continue
            parent_name, child = name.rsplit(".", 1)
            parent = self.backbone.get_submodule(parent_name)
            setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
        if not replaced:
            raise RuntimeError(f"No LoRA targets matched {targets}")
        self.replaced = replaced

    def forward(self, inputs):
        hidden = self.backbone(**inputs).last_hidden_state
        # Match the upstream FedE4RAG calculator's mean pooling.
        return F.normalize(hidden.mean(dim=1), dim=1)

    def lora_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()
                if "lora_A" in n or "lora_B" in n}

    def load_lora(self, state):
        own = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))


class PairDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def choose_positive(levels):
    # One evidence item only: this is deliberately not a hierarchical objective.
    for level in ("row", "table", "cell"):
        ids = levels.get(level, [])
        if ids:
            return sorted(ids)[0]
    return None


def load_pairs(variant, partition, selected_clients, limit, seed):
    base = Path("data/processed")
    wanted_queries = set()
    q_client = {}
    for cid in selected_clients:
        ids = partition["clients"][str(cid)]["query_ids"]["train"]
        if limit:
            ids = sorted(ids)
            rng = random.Random(seed + cid)
            rng.shuffle(ids)
            ids = ids[:limit]
        for qid in ids:
            wanted_queries.add(qid); q_client[qid] = cid
    questions = {}
    for item in jsonl(base / variant / "queries_train.jsonl"):
        if item["query_id"] in wanted_queries:
            questions[item["query_id"]] = item["question"]
    positive_ids = {}
    for item in jsonl(base / "evidence_map.jsonl"):
        qid = item["query_id"]
        if qid in wanted_queries:
            positive_ids[qid] = choose_positive(
                item["variants"][variant]["positive_evidence_ids"])
    needed = {x for x in positive_ids.values() if x}
    texts = {}
    for item in jsonl(base / variant / "corpus_train.jsonl"):
        if item["evidence_id"] in needed:
            texts[item["evidence_id"]] = item["text"]
    result = {cid: [] for cid in selected_clients}
    missing = []
    for qid in sorted(wanted_queries):
        eid = positive_ids.get(qid)
        if qid not in questions or eid not in texts:
            missing.append(qid); continue
        result[q_client[qid]].append((questions[qid], texts[eid]))
    if missing:
        logging.warning("Skipped %d queries without a resolved positive", len(missing))
    return result


def collate(tokenizer, max_length):
    def fn(batch):
        q, p = zip(*batch)
        qi = tokenizer(list(q), padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt")
        pi = tokenizer(list(p), padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt")
        return qi, pi
    return fn


def train_client(student, teacher, pairs, cfg, cid, round_no, device):
    student.train(); teacher.eval()
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))
    loader = DataLoader(PairDataset(pairs), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"] + 1000 * round_no + cid),
                        collate_fn=collate(cfg["_tokenizer"], cfg["max_length"]))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["precision"] == "fp16")
    optimizer.zero_grad(set_to_none=True)
    losses, ce_values, kd_values = [], [], []
    teacher_detached = teacher_no_grad = True
    grad_finite = True
    fp16_overflow_skips = 0
    steps = 0
    started = time.monotonic()
    for epoch in range(cfg["local_epochs"]):
        for batch_idx, (qi, pi) in enumerate(loader):
            qi = {k: v.to(device) for k, v in qi.items()}
            pi = {k: v.to(device) for k, v in pi.items()}
            amp = torch.cuda.amp.autocast if cfg["precision"] == "fp16" else nullcontext
            with torch.inference_mode():
                with amp():
                    tq, tp = teacher(qi), teacher(pi)
                    teacher_logits = tq @ tp.T
            teacher_detached &= not teacher_logits.requires_grad
            teacher_no_grad &= all(p.grad is None for p in teacher.parameters())
            with amp():
                sq, sp = student(qi), student(pi)
                logits = sq @ sp.T
                labels = torch.arange(logits.shape[0], device=device)
                ce = F.cross_entropy(logits / float(cfg["temperature"]), labels)
                kd = F.mse_loss(logits, teacher_logits.detach())
                loss = (ce + float(cfg["teacher_mse_weight"]) * kd) / cfg["grad_accum"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss client={cid} round={round_no}")
            scaler.scale(loss).backward()
            boundary = (batch_idx + 1) % cfg["grad_accum"] == 0 or batch_idx + 1 == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                raw_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                                 for p in trainable)
                if not raw_finite and scaler.is_enabled():
                    # Dynamic loss scaling intentionally skips this update. No
                    # non-finite value is ever applied to model parameters.
                    fp16_overflow_skips += 1
                    logging.warning("FP16 overflow skipped client=%d round=%d batch=%d",
                                    cid, round_no, batch_idx)
                elif not raw_finite:
                    grad_finite = False
                    raise FloatingPointError(f"non-finite gradient client={cid} round={round_no}")
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                steps += 1
            losses.append(loss.item() * cfg["grad_accum"])
            ce_values.append(ce.item()); kd_values.append(kd.item())
    return student.lora_state(), {
        "client": cid, "examples": len(pairs), "optimizer_steps": steps,
        "loss": float(np.mean(losses)), "ce_loss": float(np.mean(ce_values)),
        "teacher_mse": float(np.mean(kd_values)), "teacher_logits_detached": teacher_detached,
        "teacher_no_grad": teacher_no_grad, "grad_finite": grad_finite,
        "fp16_overflow_skips": fp16_overflow_skips,
        "seconds": time.monotonic() - started,
    }


def aggregate(states, weights):
    total = float(sum(weights))
    return {name: sum(state[name].float() * (w / total) for state, w in zip(states, weights))
            for name in states[0]}


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variant", choices=["flat", "predicted_structure"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    seed_all(cfg["seed"])
    if cfg["save"] != "best_and_last":
        raise ValueError("Day-5 policy requires save: best_and_last")
    if not torch.cuda.is_available() and str(cfg["device"]).startswith("cuda"):
        raise RuntimeError("CUDA is required by this configuration")
    device = torch.device(cfg["device"])
    partition = json.loads(Path(cfg["partition"]).read_text())
    selected = list(range(cfg["clients"]))
    pairs = load_pairs(args.variant, partition, selected,
                       cfg.get("smoke_max_examples_per_client"), cfg["seed"])
    if any(not pairs[c] for c in selected):
        raise RuntimeError("one or more clients have no training pairs")
    run_dir = Path(cfg["output_root"]) / args.run_name
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg_out.update({"variant": args.variant, "run_name": args.run_name,
                    "partition_sha256": partition["mapping_sha256"]})
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg_out, indent=2, sort_keys=True) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    cfg["_tokenizer"] = tokenizer
    logging.info("Loading student and frozen global teacher from %s", cfg["model"])
    student = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                      cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
    teacher = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                      cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
    global_state = student.lora_state()
    lora_bytes = sum(x.numel() * x.element_size() for x in global_state.values())
    start_round, best_loss = 1, float("inf")
    last_path = ckpt_dir / "last.pt"
    if cfg.get("resume") and last_path.exists():
        saved = torch.load(last_path, map_location="cpu")
        global_state = saved["lora_state"]; start_round = saved["round"] + 1
        best_loss = saved["best_loss"]
        logging.info("Resuming after round %d", saved["round"])
    fields = ["round", "loss", "ce_loss", "teacher_mse", "examples", "seconds",
              "peak_gpu_mib", "round_communication_bytes"]
    curve_path = run_dir / "convergence.csv"
    if start_round == 1:
        with curve_path.open("w", newline="") as f: csv.DictWriter(f, fields).writeheader()
    run_started = time.monotonic()
    audit = {"teacher_logits_detached": True, "teacher_no_grad": True, "grad_finite": True,
             "clients_sequential": True, "max_concurrent_clients": 1,
             "checkpoint_policy": "best_and_last", "client_events": []}
    for round_no in range(start_round, cfg["rounds"] + 1):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        states, weights, client_metrics = [], [], []
        round_started = time.monotonic()
        teacher.load_lora(global_state); teacher.eval()
        for cid in selected:  # Intentional sequential execution.
            logging.info("ROUND %d CLIENT %d START", round_no, cid)
            audit["client_events"].append({"round": round_no, "client": cid, "event": "start"})
            student.load_lora(global_state)
            state, metrics = train_client(student, teacher, pairs[cid], cfg, cid, round_no, device)
            states.append(state); weights.append(len(pairs[cid])); client_metrics.append(metrics)
            for key in ("teacher_logits_detached", "teacher_no_grad", "grad_finite"):
                audit[key] &= metrics[key]
            audit["client_events"].append({"round": round_no, "client": cid, "event": "end"})
            logging.info("ROUND %d CLIENT %d END loss=%.6f sec=%.1f",
                         round_no, cid, metrics["loss"], metrics["seconds"])
        global_state = aggregate(states, weights)
        avg = lambda key: float(np.average([m[key] for m in client_metrics], weights=weights))
        row = {"round": round_no, "loss": avg("loss"), "ce_loss": avg("ce_loss"),
               "teacher_mse": avg("teacher_mse"), "examples": sum(weights),
               "seconds": time.monotonic() - round_started,
               "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                if torch.cuda.is_available() else 0),
               "round_communication_bytes": 2 * len(selected) * lora_bytes}
        with curve_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fields); writer.writerow(row)
        payload = {"round": round_no, "best_loss": min(best_loss, row["loss"]),
                   "lora_state": global_state, "config": cfg_out}
        atomic_save(payload, last_path)
        if row["loss"] < best_loss:
            best_loss = row["loss"]; payload["best_loss"] = best_loss
            atomic_save(payload, ckpt_dir / "best.pt")
        logging.info("ROUND %d COMPLETE loss=%.6f peak=%.1fMiB comm=%d",
                     round_no, row["loss"], row["peak_gpu_mib"], row["round_communication_bytes"])
    checkpoints = sorted(p.name for p in ckpt_dir.iterdir() if p.is_file())
    audit.update({
        "accepted": all(audit[k] for k in ("teacher_logits_detached", "teacher_no_grad",
                                           "grad_finite", "clients_sequential"))
                    and checkpoints == ["best.pt", "last.pt"],
        "checkpoints": checkpoints, "lora_parameter_bytes": lora_bytes,
        "total_communication_bytes": 2 * len(selected) * lora_bytes * cfg["rounds"],
        "total_seconds": time.monotonic() - run_started,
        "completed_rounds": cfg["rounds"],
    })
    (run_dir / "training_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    logging.info("TRAINING COMPLETE audit=%s", audit["accepted"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("training failed")
        sys.exit(1)
