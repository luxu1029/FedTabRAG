#!/usr/bin/env python3
"""Exact multi-positive retrieval evaluation over frozen BGE embeddings."""

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


LOG = logging.getLogger("evaluate_retrieval")
LEVELS = ("table", "row", "cell")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--structure-per-sample", type=Path, required=True)
    parser.add_argument("--k", default="1,5,10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--score-chunk-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--structure-split",
        choices=("auto", "dev", "test"),
        default="auto",
        help="Structure-metric split; auto requires all queries to share one split.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def encode_queries(
    texts: Sequence[str], model_path: Path, device: torch.device, max_length: int, batch_size: int
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = []
    for start in range(0, len(texts), batch_size):
        batch = [QUERY_INSTRUCTION + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            values = model(**encoded).last_hidden_state[:, 0]
        output.append(F.normalize(values.float(), p=2, dim=1).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(output).astype(np.float32)


def ideal_dcg(relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant, k) + 1))


def metrics_from_ranks(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float]:
    ordered = sorted(ranks)
    result = {f"recall@{k}": float(any(rank <= k for rank in ordered)) for k in ks}
    result["mrr"] = 1.0 / ordered[0] if ordered else 0.0
    result["ap"] = (
        sum(index / rank for index, rank in enumerate(ordered, 1)) / len(ordered)
        if ordered
        else 0.0
    )
    dcg10 = sum(1.0 / math.log2(rank + 1) for rank in ordered if rank <= 10)
    denominator = ideal_dcg(len(ordered), 10)
    result["ndcg@10"] = dcg10 / denominator if denominator else 0.0
    return result


def mean_dict(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    return {key: statistics.fmean(item[key] for item in items) for key in items[0]}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    state = json.loads((args.index / "state.json").read_text(encoding="utf-8"))
    if not state.get("complete"):
        raise ValueError("Index is incomplete")
    embeddings = np.load(args.index / "embeddings.npy", mmap_mode="r")
    corpus = read_jsonl(args.corpus)
    queries = read_jsonl(args.queries)
    if not queries:
        raise ValueError("Queries are empty")
    if len(corpus) != embeddings.shape[0]:
        raise ValueError("Corpus/index count mismatch")
    variant = str(queries[0]["source_variant"])
    maps = {item["query_id"]: item for item in read_jsonl(args.evidence_map)}
    query_splits = {str(item.get("split")) for item in queries}
    if args.structure_split == "auto":
        if len(query_splits) != 1 or "None" in query_splits:
            raise ValueError(f"Cannot infer one structure split from queries: {query_splits}")
        structure_split = next(iter(query_splits))
    else:
        structure_split = args.structure_split
    if query_splits != {structure_split}:
        raise ValueError(
            f"Query split {sorted(query_splits)} does not match structure split {structure_split}"
        )
    structure = {
        item["sample_key"]: float(item["s_teds"])
        for item in read_jsonl(args.structure_per_sample)
        if item.get("split") == structure_split
    }
    missing_structure = sorted({item["sample_key"] for item in queries} - structure.keys())
    if missing_structure:
        raise KeyError(
            f"Missing structure metrics for {len(missing_structure)} queries; first={missing_structure[0]}"
        )
    s_values = sorted(structure.values())
    quartiles = [float(np.quantile(s_values, q)) for q in (0.25, 0.5, 0.75)]

    def bucket(value: float) -> str:
        if value <= quartiles[0]:
            return "q1_low"
        if value <= quartiles[1]:
            return "q2"
        if value <= quartiles[2]:
            return "q3"
        return "q4_high"

    query_embeddings = encode_queries(
        [item["question"] for item in queries],
        args.model,
        device,
        int(state["max_length"]),
        args.query_batch_size,
    )
    ks = sorted({int(value) for value in args.k.split(",")})
    per_query = [
        {
            "query_id": item["query_id"],
            "sample_id": item["sample_id"],
            "question": item["question"],
            "s_teds": structure[item["sample_key"]],
            "s_teds_bucket": bucket(structure[item["sample_key"]]),
            "levels": {},
        }
        for item in queries
    ]
    evidence_to_index = {item["evidence_id"]: index for index, item in enumerate(corpus)}

    for level in LEVELS:
        level_indices = np.asarray(
            [index for index, item in enumerate(corpus) if item["level"] == level], dtype=np.int64
        )
        global_to_level = {int(global_index): local for local, global_index in enumerate(level_indices)}
        LOG.info("Evaluating variant=%s level=%s corpus=%d", variant, level, len(level_indices))
        for q_start in range(0, len(queries), args.query_batch_size):
            q_end = min(len(queries), q_start + args.query_batch_size)
            batch_q = torch.from_numpy(query_embeddings[q_start:q_end]).to(device)
            scores = np.empty((q_end - q_start, len(level_indices)), dtype=np.float32)
            for c_start in range(0, len(level_indices), args.score_chunk_size):
                c_end = min(len(level_indices), c_start + args.score_chunk_size)
                block = np.asarray(embeddings[level_indices[c_start:c_end]], dtype=np.float32)
                block_tensor = torch.from_numpy(block).to(device)
                scores[:, c_start:c_end] = (batch_q @ block_tensor.T).cpu().numpy()
            for offset, query in enumerate(queries[q_start:q_end]):
                positive_ids = maps[query["query_id"]]["variants"][variant]["positive_evidence_ids"][level]
                positive_local = []
                for evidence_id in positive_ids:
                    global_index = evidence_to_index.get(evidence_id)
                    if global_index is not None and global_index in global_to_level:
                        positive_local.append(global_to_level[global_index])
                if not positive_local:
                    per_query[q_start + offset]["levels"][level] = {"eligible": False}
                    continue
                row_scores = scores[offset]
                positive_scores = row_scores[positive_local]
                ranks = [
                    int(
                        np.count_nonzero(row_scores > positive_score)
                        + np.count_nonzero(
                            (row_scores == positive_score)
                            & (np.arange(len(row_scores)) < positive_index)
                        )
                        + 1
                    )
                    for positive_score, positive_index in zip(positive_scores, positive_local)
                ]
                top_count = min(10, len(row_scores))
                top_local = np.argpartition(-row_scores, top_count - 1)[:top_count]
                top_local = top_local[
                    np.lexsort((top_local, -row_scores[top_local]))
                ]
                top_global = level_indices[top_local]
                level_metrics = metrics_from_ranks(ranks, ks)
                per_query[q_start + offset]["levels"][level] = {
                    "eligible": True,
                    "positive_count": len(positive_local),
                    "positive_ranks": sorted(ranks),
                    "metrics": level_metrics,
                    "top10": [
                        {
                            "evidence_id": corpus[int(global_index)]["evidence_id"],
                            "score": float(row_scores[int(local_index)]),
                            "is_positive": corpus[int(global_index)]["evidence_id"] in set(positive_ids),
                        }
                        for local_index, global_index in zip(top_local, top_global)
                    ],
                }
            LOG.info("Progress level=%s queries=%d/%d", level, q_end, len(queries))

    metrics = {}
    by_bucket = {}
    for level in LEVELS:
        eligible = [
            item["levels"][level]["metrics"]
            for item in per_query
            if item["levels"][level].get("eligible")
        ]
        metrics[level] = {"eligible_queries": len(eligible), **mean_dict(eligible)}
        by_bucket[level] = {}
        for name in ("q1_low", "q2", "q3", "q4_high"):
            values = [
                item["levels"][level]["metrics"]
                for item in per_query
                if item["s_teds_bucket"] == name and item["levels"][level].get("eligible")
            ]
            by_bucket[level][name] = {"eligible_queries": len(values), **mean_dict(values)}
    multi_positive = {}
    for level in LEVELS:
        values = [
            item["levels"][level]
            for item in per_query
            if item["levels"][level].get("eligible")
        ]
        multi_positive[level] = {
            "eligible_queries": len(values),
            "multi_positive_queries": sum(item["positive_count"] > 1 for item in values),
            "max_positive_count": max((item["positive_count"] for item in values), default=0),
        }
    report = {
        "variant": variant,
        "protocol": {
            "model": str(args.model),
            "model_fingerprint": state["model_fingerprint"],
            "pooling": state["pooling"],
            "normalize": state["normalize"],
            "max_length": state["max_length"],
            "query_instruction": QUERY_INSTRUCTION,
            "ranking": "exact_full_level_corpus",
            "map_definition": "mean exact average precision over all positives",
            "split": structure_split,
        },
        "query_count": len(queries),
        "s_teds_quartile_boundaries": quartiles,
        "metrics": metrics,
        "by_s_teds_bucket": by_bucket,
        "multi_positive_audit": multi_positive,
        "per_query": per_query,
    }
    atomic_json(args.output, report)
    LOG.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Fatal retrieval evaluation error")
        sys.exit(2)
