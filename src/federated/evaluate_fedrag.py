#!/usr/bin/env python3
"""Evaluate a trained Day-5 LoRA checkpoint globally and by frozen client."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from train_fedrag import Encoder, jsonl, seed_all

LEVELS = ("table", "row", "cell")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", choices=["best", "last"], default="best")
    p.add_argument("--variant", choices=["flat", "predicted_structure"], required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--score-chunk-size", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpus-path", type=Path,
                   help="Optional structure-perturbed corpus copy for robustness evaluation")
    return p.parse_args()


def encode(model, tokenizer, texts, device, batch_size, max_length):
    values = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            values.append(model(batch).float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def metric(ranks):
    ranks = sorted(ranks)
    if not ranks:
        return {}
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(ranks), 10) + 1))
    return {
        "recall@1": float(ranks[0] <= 1),
        "recall@5": float(ranks[0] <= 5),
        "mrr": 1 / ranks[0],
        "map": sum(i / rank for i, rank in enumerate(ranks, 1)) / len(ranks),
        "ndcg@10": sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / idcg,
    }


def mean_metrics(rows):
    return ({k: statistics.fmean(x[k] for x in rows) for k in rows[0]} if rows else {})


def main():
    args = parse_args(); seed_all(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = json.loads((args.run_dir / "resolved_config.json").read_text())
    partition = json.loads(args.partition.read_text())
    if cfg["partition_sha256"] != partition["mapping_sha256"]:
        raise ValueError("partition hash mismatch")
    device = torch.device(args.device)
    model = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                    cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
    saved = torch.load(args.run_dir / "checkpoints" / f"{args.checkpoint}.pt",
                       map_location="cpu")
    model.load_lora(saved["lora_state"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    corpus_path = (args.corpus_path or
                   Path("data/processed") / args.variant / "corpus_test.jsonl")
    corpus = list(jsonl(corpus_path))
    queries = list(jsonl(Path("data/processed") / args.variant / "queries_test.jsonl"))
    maps = {x["query_id"]: x for x in jsonl(Path("data/processed/evidence_map.jsonl"))
            if x["split"] == "test"}
    owner = {qid: int(cid) for cid, c in partition["clients"].items()
             for qid in c["query_ids"]["test"]}
    logging.info("Encoding corpus=%d queries=%d", len(corpus), len(queries))
    corpus_emb = encode(model, tokenizer, [x["text"] for x in corpus], device,
                        args.batch_size, cfg["max_length"])
    query_emb = encode(model, tokenizer, [x["question"] for x in queries], device,
                       args.batch_size, cfg["max_length"])
    evidence_index = {x["evidence_id"]: i for i, x in enumerate(corpus)}
    per_query = {x["query_id"]: {"client": owner[x["query_id"]], "levels": {}}
                 for x in queries}
    for level in LEVELS:
        indices = np.asarray([i for i, x in enumerate(corpus) if x["level"] == level])
        reverse = {int(g): i for i, g in enumerate(indices)}
        logging.info("Scoring level=%s evidence=%d", level, len(indices))
        for qs in range(0, len(queries), args.batch_size):
            qe = query_emb[qs:qs + args.batch_size]
            scores = np.empty((len(qe), len(indices)), np.float32)
            for cs in range(0, len(indices), args.score_chunk_size):
                block = indices[cs:cs + args.score_chunk_size]
                scores[:, cs:cs + len(block)] = qe @ corpus_emb[block].T
            for off, query in enumerate(queries[qs:qs + len(qe)]):
                positive_ids = maps[query["query_id"]]["variants"][args.variant][
                    "positive_evidence_ids"][level]
                positive = [reverse[evidence_index[e]] for e in positive_ids
                            if e in evidence_index and evidence_index[e] in reverse]
                if not positive:
                    continue
                row = scores[off]
                ranks = [int(np.count_nonzero(row > row[p]) +
                             np.count_nonzero((row == row[p]) & (np.arange(len(row)) < p)) + 1)
                         for p in positive]
                per_query[query["query_id"]]["levels"][level] = metric(ranks)
    global_metrics, clients = {}, {}
    for level in LEVELS:
        rows = [x["levels"][level] for x in per_query.values() if level in x["levels"]]
        global_metrics[level] = {"eligible_queries": len(rows), **mean_metrics(rows)}
        client_rows = {}
        for cid in range(cfg["clients"]):
            vals = [x["levels"][level] for x in per_query.values()
                    if x["client"] == cid and level in x["levels"]]
            client_rows[str(cid)] = {"eligible_queries": len(vals), **mean_metrics(vals)}
        clients[level] = client_rows
    fairness = {}
    for level in LEVELS:
        fairness[level] = {}
        keys = ("recall@1", "recall@5", "mrr", "map", "ndcg@10")
        for key in keys:
            vals = [clients[level][str(cid)].get(key, 0.0) for cid in range(cfg["clients"])]
            fairness[level][key] = {
                "macro": statistics.fmean(vals), "worst_client": min(vals),
                "std": statistics.pstdev(vals),
            }
    audit = json.loads((args.run_dir / "training_audit.json").read_text())
    curve_rows = list(csv.DictReader((args.run_dir / "convergence.csv").open()))
    report = {
        "variant": args.variant, "checkpoint": args.checkpoint,
        "checkpoint_round": saved["round"], "partition_sha256": partition["mapping_sha256"],
        "corpus_path": str(corpus_path),
        "protocol": {"pooling": "mean_matching_upstream_FedE4RAG",
                     "ranking": "exact_full_test_level_corpus",
                     "aggregation": cfg.get(
                         "aggregation", "sample_weighted_vanilla_FedAvg"),
                     "hierarchical_loss": "hierarchy" in cfg.get("features", "")},
        "global": global_metrics, "clients": clients, "fairness": fairness,
        "communication": {
            "lora_parameter_bytes": audit["lora_parameter_bytes"],
            "per_round_bytes": audit["total_communication_bytes"] // cfg["rounds"],
            "total_bytes": audit["total_communication_bytes"],
        },
        "training": {
            "total_seconds": sum(float(x["seconds"]) for x in curve_rows),
            "peak_gpu_mib": max(float(x["peak_gpu_mib"]) for x in curve_rows),
            "rounds": len(curve_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n"); tmp.replace(args.output)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("evaluation failed")
        sys.exit(1)
