#!/usr/bin/env python3
"""Freeze a deterministic, performance-blind stratified FinQA test subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finqa-test", type=Path, default=Path("data/raw/finqa/test.json"))
    parser.add_argument("--structure-metrics", type=Path,
                        default=Path("outputs/unitable/structure_metrics_per_sample.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/day9/sample_ids.json"))
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def table_bin(table: list[list[str]]) -> str:
    cells = sum(len(row) for row in table)
    return "small" if cells <= 20 else "medium" if cells <= 50 else "large"


def quality_bin(score: float) -> str:
    return "low" if score < 0.90 else "medium" if score < 0.95 else "high"


def allocate(groups: dict[tuple[str, str, str], list[dict]], size: int) -> dict:
    total = sum(len(rows) for rows in groups.values())
    exact = {key: size * len(rows) / total for key, rows in groups.items()}
    allocation = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - int(exact[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate requested sample size")
    return allocation


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen subset: {args.output}")
    data = json.loads(args.finqa_test.read_text())
    structure = {}
    with args.structure_metrics.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "test":
                structure[row["sample_key"]] = float(row["s_teds"])
    rows = []
    for index, item in enumerate(data):
        query_id = f"test:{item['id']}:{index}"
        if query_id not in structure:
            raise KeyError(f"missing structure metric: {query_id}")
        program = str(item["qa"].get("program") or "unknown")
        program_type = program.split("(", 1)[0].strip() or "unknown"
        rows.append({
            "query_id": query_id, "sample_id": item["id"],
            "program_type": program_type, "table_size_bin": table_bin(item["table"]),
            "structure_quality_bin": quality_bin(structure[query_id]),
            "s_teds": structure[query_id],
        })
    groups = defaultdict(list)
    for row in rows:
        groups[(row["program_type"], row["table_size_bin"],
                row["structure_quality_bin"])].append(row)
    allocation = allocate(groups, args.size)
    selected = []
    for offset, key in enumerate(sorted(groups)):
        candidates = sorted(groups[key], key=lambda row: row["query_id"])
        random.Random(args.seed + offset).shuffle(candidates)
        selected.extend(candidates[:allocation[key]])
    selected.sort(key=lambda row: row["query_id"])
    if len(selected) != args.size or len({x["query_id"] for x in selected}) != args.size:
        raise AssertionError("frozen subset cardinality/uniqueness failure")
    payload = {
        "seed": args.seed, "size": args.size, "source_split": "test",
        "selection_uses_method_performance": False,
        "stratification": ["program_type", "table_size_bin", "structure_quality_bin"],
        "thresholds": {"table_cells": [20, 50], "s_teds": [0.90, 0.95]},
        "finqa_test_sha256": sha256(args.finqa_test),
        "structure_metrics_sha256": sha256(args.structure_metrics),
        "distribution": {
            field: dict(Counter(row[field] for row in selected))
            for field in ("program_type", "table_size_bin", "structure_quality_bin")
        },
        "samples": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    logging.info("froze %d IDs at %s", len(selected), args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("subset freeze failed")
        raise
