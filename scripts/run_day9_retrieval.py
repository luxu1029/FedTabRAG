#!/usr/bin/env python3
"""Retrieve one frozen Day-9 method over the same 100 FinQA test questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/federated"))
from train_fedrag import Encoder, jsonl, seed_all  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=Path, default=Path("configs/day9_methods.json"))
    parser.add_argument("--sample-ids", type=Path, default=Path("outputs/day9/sample_ids.json"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day9/retrieval"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encode(model, tokenizer, texts, device, batch_size, max_length):
    blocks = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            blocks.append(model(batch).float().cpu().numpy())
    return np.concatenate(blocks).astype(np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(args.seed)
    registry = json.loads(args.methods.read_text())
    if args.method not in registry["methods"]:
        raise KeyError(args.method)
    spec = registry["methods"][args.method]
    protocol = registry["protocol"]
    if args.seed != protocol["seed"]:
        raise ValueError("seed differs from frozen protocol")
    output = args.output_dir / f"{args.method}.jsonl"
    state_path = args.output_dir / f"{args.method}.state.json"
    if output.exists() or state_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.method} retrieval")
    sample = json.loads(args.sample_ids.read_text())
    selected = {row["query_id"] for row in sample["samples"]}
    variant = spec["variant"]
    corpus_path = Path("data/processed") / variant / "corpus_test.jsonl"
    query_path = Path("data/processed") / variant / "queries_test.jsonl"
    corpus = list(jsonl(corpus_path))
    queries = [row for row in jsonl(query_path) if row["query_id"] in selected]
    queries.sort(key=lambda row: row["query_id"])
    if len(queries) != sample["size"]:
        raise ValueError("selected query coverage mismatch")
    checkpoint_path = None
    if spec.get("run"):
        run = Path("outputs/federated") / spec["run"]
        cfg = json.loads((run / "resolved_config.json").read_text())
        checkpoint_path = run / spec.get("checkpoint", "checkpoints/best.pt")
    else:
        cfg = {
            "model": "models/bge-base-en-v1.5", "lora_rank": 8, "lora_alpha": 16,
            "lora_dropout": 0.05, "lora_targets": ["query", "value"],
            "max_length": 256,
        }
    device = torch.device(args.device)
    model = Encoder(cfg["model"], cfg["lora_rank"], cfg["lora_alpha"],
                    cfg["lora_dropout"], set(cfg["lora_targets"])).to(device)
    checkpoint_sha = None
    checkpoint_round = None
    if checkpoint_path is not None:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_lora(saved["lora_state"])
        checkpoint_sha = sha256(checkpoint_path)
        checkpoint_round = saved["round"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    logging.info("encoding method=%s corpus=%d queries=%d", args.method,
                 len(corpus), len(queries))
    corpus_emb = encode(model, tokenizer, [row["text"] for row in corpus], device,
                        args.batch_size, cfg["max_length"])
    query_emb = encode(model, tokenizer, [row["question"] for row in queries], device,
                       args.batch_size, cfg["max_length"])
    maps = {row["query_id"]: row for row in jsonl(Path("data/processed/evidence_map.jsonl"))
            if row["split"] == "test" and row["query_id"] in selected}
    top_k = int(protocol["top_k"])
    hits, reciprocal = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for start in range(0, len(queries), args.batch_size):
            scores = query_emb[start:start + args.batch_size] @ corpus_emb.T
            width = min(top_k, len(corpus))
            candidate = np.argpartition(-scores, width - 1, axis=1)[:, :width]
            for offset, query in enumerate(queries[start:start + len(scores)]):
                indices = sorted(candidate[offset], key=lambda index: (-scores[offset, index], index))
                positive_by_level = maps[query["query_id"]]["variants"][variant][
                    "positive_evidence_ids"]
                positives = set().union(*positive_by_level.values())
                ranks = [rank for rank, index in enumerate(indices, 1)
                         if corpus[index]["evidence_id"] in positives]
                hit = bool(ranks)
                hits.append(float(hit))
                reciprocal.append(1.0 / min(ranks) if ranks else 0.0)
                row = {
                    "method": args.method, "variant": variant,
                    "query_id": query["query_id"], "question": query["question"],
                    "top_k": top_k, "gold_hit": hit,
                    "first_gold_rank": min(ranks) if ranks else None,
                    "gold_evidence_ids": sorted(positives),
                    "retrieved": [
                        {"rank": rank, "score": float(scores[offset, index]),
                         "evidence_id": corpus[index]["evidence_id"],
                         "level": corpus[index]["level"], "text": corpus[index]["text"]}
                        for rank, index in enumerate(indices, 1)
                    ],
                }
                handle.write(json.dumps(row) + "\n")
    state = {
        "method": args.method, "variant": variant, "note": spec.get("note"),
        "seed": args.seed, "top_k": top_k, "sample_ids_sha256": sha256(args.sample_ids),
        "corpus_sha256": sha256(corpus_path), "queries_sha256": sha256(query_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_round": checkpoint_round,
        "queries": len(queries), "corpus_items": len(corpus),
        "unified_evidence_recall@5": float(np.mean(hits)),
        "unified_evidence_mrr@5": float(np.mean(reciprocal)),
        "output_sha256": sha256(output), "uses_test_for_selection": False,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    logging.info("wrote %s recall@5=%.4f", output, state["unified_evidence_recall@5"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-9 retrieval failed")
        raise
