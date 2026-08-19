#!/usr/bin/env python3
"""Day-6 FedTabRAG hierarchy/hard-negative training with vanilla FedAvg."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fedtabrag_losses import (LEVELS, hierarchical_kd, hierarchical_loss,
                              kd_weight_for_round, margin_hard_negative_loss)
from train_fedrag import Encoder, aggregate, atomic_save, jsonl, seed_all
from dev_evaluation import DevEvaluator
from local_queue import audit as audit_local_queue, empty_queue, push as queue_push
from aggregation_ru import (AGGREGATIONS, aggregation_weights, client_reliability,
                            quality_values, sanitize_utilities)


class Examples(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def load_examples(partition, clients, hard_path=None, limit=None, seed=42):
    wanted = {}
    for cid in clients:
        ids = list(partition["clients"][str(cid)]["query_ids"]["train"])
        if limit:
            random.Random(seed + cid).shuffle(ids); ids = ids[:limit]
        wanted.update({qid: cid for qid in ids})
    rows = {}
    for x in jsonl(Path("data/processed/predicted_structure/queries_train.jsonl")):
        if x["query_id"] in wanted:
            rows[x["query_id"]] = {"query_id": x["query_id"], "question": x["question"],
                                   "client": wanted[x["query_id"]], "positive_ids": {}}
    needed = set()
    for x in jsonl(Path("data/processed/evidence_map.jsonl")):
        if x["query_id"] in rows:
            pos = x["variants"]["predicted_structure"]["positive_evidence_ids"]
            rows[x["query_id"]]["positive_ids"] = {k: list(v) for k, v in pos.items()}
            needed.update(e for vals in pos.values() for e in vals)
    hard = {}
    if hard_path:
        for x in jsonl(Path(hard_path)):
            if x["query_id"] in rows:
                hard[x["query_id"]] = x["negatives"]
                needed.update(n["evidence_id"] for n in x["negatives"])
    texts = {}
    for x in jsonl(Path("data/processed/predicted_structure/corpus_train.jsonl")):
        if x["evidence_id"] in needed: texts[x["evidence_id"]] = x["text"]
    by_client = {cid: [] for cid in clients}
    for qid in sorted(rows):
        row = rows[qid]
        row["texts"] = texts
        row["hard"] = hard.get(qid, [])
        if row["positive_ids"].get("table"):
            by_client[row["client"]].append(row)
    return by_client


def tokenize(tokenizer, texts, cfg, device):
    x = tokenizer(texts, padding=True, truncation=True, max_length=cfg["max_length"],
                  return_tensors="pt")
    return {k: v.to(device) for k, v in x.items()}


def level_batch(model, query_emb, batch, level, tokenizer, cfg, device,
                teacher=None, teacher_query_emb=None, queue=None):
    candidates = sorted({eid for row in batch for eid in row["positive_ids"].get(level, [])})
    if not candidates:
        dummy = query_emb @ query_emb[:1].T
        return dummy, torch.zeros_like(dummy, dtype=torch.bool), None, None, dummy.detach()
    evidence_emb = model(tokenize(tokenizer, [batch[0]["texts"][e] for e in candidates],
                                  cfg, device))
    teacher_evidence = evidence_emb.detach()
    if teacher is not None:
        with torch.inference_mode():
            teacher_evidence = teacher(tokenize(
                tokenizer, [batch[0]["texts"][e] for e in candidates], cfg, device))
    all_ids = list(candidates)
    all_student = evidence_emb
    all_teacher = teacher_evidence
    if queue is not None and queue[level]["ids"]:
        all_ids += queue[level]["ids"]
        all_student = torch.cat([evidence_emb, torch.stack(queue[level]["student"])], dim=0)
        all_teacher = torch.cat([teacher_evidence, torch.stack(queue[level]["teacher"])], dim=0)
    logits = query_emb @ all_student.T / float(cfg["temperature"])
    mask = torch.tensor([[eid in set(row["positive_ids"].get(level, []))
                          for eid in all_ids] for row in batch],
                        dtype=torch.bool, device=device)
    pos_sim = logits.masked_select(mask).mean() if mask.any() else None
    neg_sim = logits.masked_select(~mask).mean() if (~mask).any() else None
    teacher_logits = logits.detach()
    if teacher is not None:
        teacher_logits = teacher_query_emb @ all_teacher.T / float(cfg["temperature"])
    if queue is not None:
        # Refresh duplicates and keep a bounded FIFO. Both branches are detached,
        # so no graph or raw text survives the batch.
        for eid, se, te in zip(candidates, evidence_emb, teacher_evidence):
            queue_push(queue[level], eid, se, te, batch[0]["client"],
                       int(cfg.get("queue_size", 0)))
    return logits, mask, pos_sim, neg_sim, teacher_logits.detach()


def grad_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None))


def train_client(model, teacher, rows, cfg, cid, round_no, hard_enabled, kd_enabled,
                 stage, device, queue_enabled=False):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]))
    fixed = stage == "single_batch"
    loader = DataLoader(Examples(rows), batch_size=(len(rows) if fixed else cfg["batch_size"]),
                        shuffle=not fixed, collate_fn=lambda x: x,
                        generator=torch.Generator().manual_seed(cfg["seed"] + cid + round_no * 1000))
    batches = [next(iter(loader))] * 50 if fixed else list(loader)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["precision"] == "fp16")
    opt.zero_grad(set_to_none=True)
    history = []; sums = defaultdict(list); counts = defaultdict(int)
    gradient_probe = {}; overflow_skips = 0
    started = time.monotonic()
    queue = {level: empty_queue() for level in LEVELS} if queue_enabled else None
    queue_candidates = 0
    for bi, batch in enumerate(batches):
        amp = torch.cuda.amp.autocast if cfg["precision"] == "fp16" else nullcontext
        with amp():
            qtoks = tokenize(cfg["_tokenizer"], [x["question"] for x in batch], cfg, device)
            qemb = model(qtoks)
            teacher_q = None
            if kd_enabled:
                with torch.inference_mode(): teacher_q = teacher(qtoks)
            logits, teacher_logits, masks, pos_sims, neg_sims = {}, {}, {}, {}, {}
            for level in LEVELS:
                (logits[level], masks[level], pos_sims[level], neg_sims[level],
                 teacher_logits[level]) = level_batch(
                    model, qemb, batch, level, cfg["_tokenizer"], cfg, device,
                    teacher if kd_enabled else None, teacher_q, queue)
                if queue is not None:
                    queue_candidates += max(0, logits[level].shape[1] -
                                            len({e for row in batch for e in
                                                 row["positive_ids"].get(level, [])}))
            hier, sub, valid = hierarchical_loss(logits, masks, cfg["hierarchy_weights"])
            hard_loss = qemb.sum() * 0.0
            hard_hits = 0
            if hard_enabled:
                hq, hp, hn = [], [], []
                for i, row in enumerate(batch):
                    if not row["hard"]: continue
                    neg = row["hard"][(round_no + bi + i) % len(row["hard"])]
                    same = row["positive_ids"].get(neg["level"], [])
                    pos_ids = same or next((row["positive_ids"][x] for x in LEVELS
                                            if row["positive_ids"].get(x)), [])
                    if not pos_ids: continue
                    hq.append(i); hp.append(row["texts"][pos_ids[0]]); hn.append(neg["text"])
                if hq:
                    pemb = model(tokenize(cfg["_tokenizer"], hp, cfg, device))
                    nemb = model(tokenize(cfg["_tokenizer"], hn, cfg, device))
                    hard_loss = margin_hard_negative_loss(qemb[hq], pemb, nemb,
                                                          cfg["hard_margin"])
                    hard_hits = len(hq)
            kd_loss = qemb.sum() * 0.0
            kd_sub = {x: kd_loss for x in LEVELS}
            kd_weight = kd_weight_for_round(
                round_no, cfg["kd_target_weight"], cfg["kd_start_round"],
                cfg["kd_warmup_rounds"]) if kd_enabled else 0.0
            if kd_enabled:
                kd_loss, kd_sub, _ = hierarchical_kd(
                    logits, teacher_logits, masks, cfg["hierarchy_weights"],
                    cfg["kd_temperature"])
            total = (hier + (float(cfg["hard_weight"]) * hard_loss if hard_enabled else 0)
                     + kd_weight * kd_loss)
            scaled = total / (1 if fixed else cfg["grad_accum"])
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss c={cid} r={round_no}")
        if not gradient_probe:
            gradient_probe = {f"grad_norm_{level}": grad_norm(sub[level], params)
                              for level in LEVELS}
            gradient_probe["grad_norm_hard"] = grad_norm(hard_loss, params) if hard_enabled else 0.0
            gradient_probe["grad_norm_kd"] = grad_norm(kd_loss, params) if kd_enabled else 0.0
        scaler.scale(scaled).backward()
        boundary = fixed or (bi + 1) % cfg["grad_accum"] == 0 or bi + 1 == len(batches)
        if boundary:
            scaler.unscale_(opt)
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
            if not finite and scaler.is_enabled():
                overflow_skips += 1
            elif not finite:
                raise FloatingPointError("non-finite fp32 gradient")
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        history.append(float(total.detach()))
        sums["loss"].append(float(total.detach())); sums["hier_loss"].append(float(hier.detach()))
        sums["hard_loss"].append(float(hard_loss.detach()))
        sums["kd_loss"].append(float(kd_loss.detach()))
        sums["kd_weight"].append(float(kd_weight))
        for level in LEVELS:
            sums[f"{level}_loss"].append(float(sub[level].detach()))
            counts[f"{level}_valid"] += valid[level]
            if pos_sims[level] is not None: sums[f"{level}_pos_sim"].append(float(pos_sims[level].detach()))
            if neg_sims[level] is not None: sums[f"{level}_neg_sim"].append(float(neg_sims[level].detach()))
            sums[f"kd_{level}_loss"].append(float(kd_sub[level].detach()))
        counts["hard_hits"] += hard_hits
    metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in sums.items()}
    metrics.update(counts); metrics.update(gradient_probe)
    metrics.update({"client": cid, "examples": len(rows), "seconds": time.monotonic()-started,
                    "fp16_overflow_skips": overflow_skips, "loss_first": history[0],
                   "loss_last": history[-1],
                    "teacher_eval": bool(teacher is None or not teacher.training),
                    "teacher_no_grad": bool(teacher is None or
                                            all(p.grad is None for p in teacher.parameters())),
                    "queue_candidates": queue_candidates,
                    "queue_cross_client_violations": (0 if queue is None else
                        sum(audit_local_queue(queue[level], cid)["cross_client"]
                            for level in LEVELS)),
                    "queue_non_train_violations": (0 if queue is None else
                        sum(audit_local_queue(queue[level], cid)["non_train"]
                            for level in LEVELS)),
                    "queue_final_sizes": ({} if queue is None else
                                          {level: len(queue[level]["ids"]) for level in LEVELS})})
    return model.lora_state(), metrics


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--features", required=True,
                   choices=["hierarchy", "hierarchy,hard_negative",
                            "hierarchy,hard_negative,hierarchical_kd",
                            "hierarchy,hard_negative,hierarchical_kd,local_queue"])
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", choices=["single_batch", "single_client", "two_client",
                                      "day75_smoke", "day75_lr_pilot",
                                      "day8_smoke", "full"],
                   default="full")
    p.add_argument("--queue-size", type=int, default=0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--rounds", type=int)
    p.add_argument("--select-by", choices=["train_loss", "dev_weighted_ndcg"],
                   default="train_loss")
    p.add_argument("--aggregation", choices=AGGREGATIONS, default="fedavg")
    p.add_argument("--aggregation-min-weight", type=float, default=0.05)
    p.add_argument("--aggregation-max-weight", type=float, default=0.5)
    p.add_argument("--structure-metrics", type=Path,
                   default=Path("outputs/unitable/structure_metrics_per_sample.jsonl"))
    return p.parse_args()


def main():
    args = args_parser(); cfg = yaml.safe_load(args.config.read_text()); seed_all(cfg["seed"])
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg["device"])
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    hard_enabled = "hard_negative" in args.features
    kd_enabled = "hierarchical_kd" in args.features
    queue_enabled = args.queue_size > 0
    cfg["queue_size"] = args.queue_size
    overrides = {
        "single_batch": (1, 1, 12), "single_client": (1, 1, None),
        "two_client": (2, 2, 32), "day75_smoke": (2, 5, 64),
        "day75_lr_pilot": (5, 3, None),
        "day8_smoke": (2, 2, 64),
        "full": (cfg["clients"], cfg["rounds"], None)}
    client_n, rounds, limit = overrides[args.stage]; clients = list(range(client_n))
    if args.rounds is not None:
        if args.rounds < 1:
            raise ValueError("--rounds must be positive")
        rounds = args.rounds
    partition = json.loads(Path(cfg["partition"]).read_text())
    rows = load_examples(partition, clients, cfg["hard_negatives"] if hard_enabled else None,
                         limit, cfg["seed"])
    run_dir = Path(cfg["output_root"]) / args.run_name; ck = run_dir/"checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    resolved = {**cfg, "features": args.features, "stage": args.stage, "clients": client_n,
                "rounds": rounds, "partition_sha256": partition["mapping_sha256"],
                "kd_enabled": kd_enabled, "queue_enabled": queue_enabled,
                "select_by": args.select_by,
                "aggregation": args.aggregation,
                "aggregation_min_weight": args.aggregation_min_weight,
                "aggregation_max_weight": args.aggregation_max_weight,
                "structure_metrics": str(args.structure_metrics)}
    (run_dir/"resolved_config.json").write_text(json.dumps(resolved, indent=2)+"\n")
    cfg["_tokenizer"] = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    model = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                    cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
    teacher = None
    if kd_enabled:
        teacher = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                          cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
        teacher.eval()
    global_state = model.lora_state()
    lora_bytes = sum(x.numel()*x.element_size() for x in global_state.values())
    dev_evaluator = DevEvaluator(partition, selected_clients=clients) if args.select_by == "dev_weighted_ndcg" else None
    local_dev_evaluators = (
        {cid: DevEvaluator(partition, selected_clients=[cid]) for cid in clients}
        if args.aggregation != "fedavg" else {})
    reliability_audit = client_reliability(
        partition, clients, args.structure_metrics) if args.aggregation != "fedavg" else {
            "clients": {str(cid): {"value": 1.0, "fallback": None} for cid in clients},
            "source": None, "split": "dev", "confidence_enabled": False,
            "confidence_s_teds_spearman": None,
        }
    curve=run_dir/"convergence.csv"
    start_round=1; best=(float("-inf") if dev_evaluator else float("inf"))
    if cfg.get("resume") and (ck/"last.pt").exists():
        saved=torch.load(ck/"last.pt",map_location="cpu"); global_state=saved["lora_state"]
        start_round=saved["round"]+1
        if dev_evaluator is not None and curve.exists():
            previous = list(csv.DictReader(curve.open()))
            best = max(float(row["dev_weighted_ndcg"]) for row in previous)
        else:
            best=saved.get("best_score", saved["best_loss"])
    fields = ["round","loss","hier_loss","table_loss","row_loss","cell_loss","hard_loss",
              "kd_loss","kd_table_loss","kd_row_loss","kd_cell_loss","kd_weight",
              "table_valid","row_valid","cell_valid","hard_hits","grad_norm_table",
              "grad_norm_row","grad_norm_cell","grad_norm_hard","grad_norm_kd",
              "fp16_overflow_skips",
              "seconds","peak_gpu_mib","round_communication_bytes",
              "dev_weighted_ndcg","dev_table_ndcg","dev_row_ndcg","dev_cell_ndcg",
              "dev_worst_ndcg","agg_min_weight","agg_max_weight","agg_weight_std"]
    if start_round==1:
        with curve.open("w",newline="") as f: csv.DictWriter(f,fields).writeheader()
    audit={"teacher_or_kd_present":kd_enabled,"aggregation":args.aggregation,
           "clients_sequential":True,"checkpoint_policy":"best_and_last","stage":args.stage,
           "features":args.features,"client_metrics":[],
           "reliability":reliability_audit,"aggregation_rounds":[],
           "aggregation_uses_test":False}
    for rnd in range(start_round,rounds+1):
        torch.cuda.reset_peak_memory_stats(device); states=[]; weights=[]; cms=[]; st=time.monotonic()
        if teacher is not None:
            teacher.load_lora(global_state); teacher.eval()
        for cid in clients:
            model.load_lora(global_state)
            state, met=train_client(model,teacher,rows[cid],cfg,cid,rnd,hard_enabled,
                                    kd_enabled,args.stage,device,queue_enabled)
            states.append(state);weights.append(len(rows[cid]));cms.append(met)
            audit["client_metrics"].append({"round":rnd,**met})
            logging.info("round=%d client=%d loss=%.5f table=%.5f row=%.5f cell=%.5f hard=%.5f",
                         rnd,cid,met["loss"],met["table_loss"],met["row_loss"],
                         met["cell_loss"],met["hard_loss"])
        local_utility = []
        utility_details = []
        if args.aggregation != "fedavg":
            for cid, state in zip(clients, states):
                model.load_lora(state)
                local_dev = local_dev_evaluators[cid].evaluate(
                    model, cfg["_tokenizer"], resolved, device)
                local_utility.append(local_dev["weighted_ndcg"])
                utility_details.append({
                    "client": cid, "value": local_dev["weighted_ndcg"],
                    "eligible_queries": {
                        level: local_dev["levels"][level]["eligible_queries"]
                        for level in LEVELS},
                })
        else:
            local_utility = [1.0] * len(clients)
            utility_details = [{"client": cid, "value": 1.0,
                                "fallback": "not_used_by_fedavg"} for cid in clients]
        local_utility, utility_fallbacks = sanitize_utilities(local_utility)
        reliability = [reliability_audit["clients"][str(cid)]["value"] for cid in clients]
        quality = quality_values(args.aggregation, reliability, local_utility)
        weight_audit = aggregation_weights(
            weights, quality, args.aggregation_min_weight,
            args.aggregation_max_weight)
        clipped_weights = weight_audit["clipped"]
        global_state=aggregate(states,clipped_weights)
        aggregation_record = {
            "round": rnd, "mode": args.aggregation, "clients": clients,
            "sample_counts": weights, "reliability": reliability,
            "utility": local_utility, "utility_details": utility_details,
            "utility_fallbacks": utility_fallbacks, "quality": quality,
            **weight_audit, "weight_sum": float(sum(clipped_weights)),
            "uses_test": False,
        }
        audit["aggregation_rounds"].append(aggregation_record)
        (run_dir/f"aggregation_round_{rnd}.json").write_text(
            json.dumps(aggregation_record, indent=2)+"\n")
        def avg(k): return float(np.average([m.get(k,0) for m in cms],weights=weights))
        row={k:(sum(m.get(k,0) for m in cms) if k.endswith("valid") or k in ("hard_hits","fp16_overflow_skips")
                else avg(k)) for k in fields if k not in ("round","seconds","peak_gpu_mib",
                                                           "round_communication_bytes")}
        row.update({"round":rnd,"seconds":time.monotonic()-st,
                    "peak_gpu_mib":torch.cuda.max_memory_allocated(device)/2**20,
                    "round_communication_bytes":2*client_n*lora_bytes,
                    "dev_weighted_ndcg":0.0,"dev_table_ndcg":0.0,
                    "dev_row_ndcg":0.0,"dev_cell_ndcg":0.0,"dev_worst_ndcg":0.0,
                    "agg_min_weight":min(clipped_weights),
                    "agg_max_weight":max(clipped_weights),
                    "agg_weight_std":float(np.std(clipped_weights))})
        if dev_evaluator is not None:
            model.load_lora(global_state)
            dev = dev_evaluator.evaluate(model, cfg["_tokenizer"], resolved, device)
            row["dev_weighted_ndcg"] = dev["weighted_ndcg"]
            for level in LEVELS:
                row[f"dev_{level}_ndcg"] = dev["levels"][level]["ndcg@10"]
            row["dev_worst_ndcg"] = min(
                dev["levels"][level]["worst_ndcg@10"] for level in LEVELS)
            (run_dir/f"dev_metrics_round_{rnd}.json").write_text(
                json.dumps(dev,indent=2)+"\n")
        with curve.open("a",newline="") as f: csv.DictWriter(f,fields).writerow(row)
        criterion = row["dev_weighted_ndcg"] if dev_evaluator else row["loss"]
        improved = criterion > best if dev_evaluator else criterion < best
        historical_best = max(best, criterion) if dev_evaluator else min(best, criterion)
        payload={"round":rnd,"best_loss":row["loss"],"best_score":historical_best,
                 "lora_state":global_state,
                 "config":resolved}
        atomic_save(payload,ck/"last.pt")
        if improved: best=criterion;atomic_save(payload,ck/"best.pt")
    checkpoints=sorted(x.name for x in ck.iterdir())
    audit.update({"completed_rounds":rounds,"checkpoints":checkpoints,
                  "lora_parameter_bytes":lora_bytes,
                  "total_communication_bytes":2*client_n*lora_bytes*rounds})
    queue_boundary_ok = all(m["queue_cross_client_violations"] == 0
                            and m["queue_non_train_violations"] == 0
                            for m in audit["client_metrics"])
    audit["queue_boundary_ok"] = queue_boundary_ok
    if dev_evaluator is not None:
        dev_rows = list(csv.DictReader(curve.open()))
        scores = [float(x["dev_weighted_ndcg"]) for x in dev_rows]
        table = [float(x["dev_table_ndcg"]) for x in dev_rows]
        row_or_cell_improved = (
            max(float(x["dev_row_ndcg"]) for x in dev_rows[1:]) > float(dev_rows[0]["dev_row_ndcg"])
            or max(float(x["dev_cell_ndcg"]) for x in dev_rows[1:]) >
            float(dev_rows[0]["dev_cell_ndcg"])) if len(dev_rows)>1 else False
        consecutive = any(scores[i] >= scores[i-1] and scores[i-1] >= scores[i-2]
                          for i in range(2,len(scores)))
        gate = {"best_score_exceeds_round1": max(scores)>scores[0],
                "two_consecutive_non_decreasing":consecutive,
                "row_or_cell_improved":row_or_cell_improved,
                "table_not_degraded_over_10pct":max(table)>=0.9*table[0],
                "queue_boundary_ok":queue_boundary_ok}
        gate["passed"]=all(gate.values())
        (run_dir/"day75_gate.json").write_text(json.dumps(gate,indent=2)+"\n")
    if args.stage=="single_batch":
        m=audit["client_metrics"][-1]; audit["overfit_loss_decreased"]=m["loss_last"]<m["loss_first"]
    audit["accepted"]=(checkpoints==["best.pt","last.pt"]
                       and (not kd_enabled or all(m["teacher_eval"] and m["teacher_no_grad"]
                                                  for m in audit["client_metrics"]))
                       and queue_boundary_ok
                       and all(math.isfinite(m["loss"]) for m in audit["client_metrics"])
                       and (args.stage!="single_batch" or audit["overfit_loss_decreased"]))
    (run_dir/"training_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    logging.info("complete accepted=%s",audit["accepted"])


if __name__=="__main__":
    try: main()
    except Exception:
        logging.exception("training failed");sys.exit(1)
