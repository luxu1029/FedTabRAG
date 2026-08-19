#!/usr/bin/env python3
"""Merge zero-shot results and extract three requested error-case groups."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--cases-per-group", type=int, default=30)
    return parser.parse_args()


def score(item: Dict[str, Any]) -> float:
    values = [
        level["metrics"]["ndcg@10"]
        for level in item["levels"].values()
        if level.get("eligible")
    ]
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    by_variant = {report["variant"]: report for report in reports}
    order = ("flat", "oracle_structure", "predicted_structure")
    if set(by_variant) != set(order):
        raise ValueError(f"Expected variants {order}, got {tuple(by_variant)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant", "level", "eligible_queries", "recall@1", "recall@5",
                "mrr", "map", "ndcg@10",
            ],
        )
        writer.writeheader()
        for variant in order:
            for level, metrics in by_variant[variant]["metrics"].items():
                writer.writerow(
                    {
                        "variant": variant,
                        "level": level,
                        "eligible_queries": metrics["eligible_queries"],
                        "recall@1": metrics["recall@1"],
                        "recall@5": metrics["recall@5"],
                        "mrr": metrics["mrr"],
                        "map": metrics["ap"],
                        "ndcg@10": metrics["ndcg@10"],
                    }
                )
    per_variant = {
        variant: {item["query_id"]: item for item in by_variant[variant]["per_query"]}
        for variant in order
    }
    ids = sorted(set.intersection(*(set(values) for values in per_variant.values())))
    comparisons = []
    for query_id in ids:
        flat = per_variant["flat"][query_id]
        oracle = per_variant["oracle_structure"][query_id]
        predicted = per_variant["predicted_structure"][query_id]
        comparisons.append(
            {
                "query_id": query_id,
                "sample_id": flat["sample_id"],
                "question": flat["question"],
                "s_teds": flat["s_teds"],
                "scores": {
                    "flat": score(flat),
                    "oracle_structure": score(oracle),
                    "predicted_structure": score(predicted),
                },
                "results": {
                    "flat": flat["levels"],
                    "oracle_structure": oracle["levels"],
                    "predicted_structure": predicted["levels"],
                },
            }
        )
    groups = {
        "unitable_better_than_flat": sorted(
            comparisons,
            key=lambda item: item["scores"]["predicted_structure"] - item["scores"]["flat"],
            reverse=True,
        ),
        "flat_better_than_unitable": sorted(
            comparisons,
            key=lambda item: item["scores"]["flat"] - item["scores"]["predicted_structure"],
            reverse=True,
        ),
        "gt_better_than_unitable": sorted(
            comparisons,
            key=lambda item: item["scores"]["oracle_structure"] - item["scores"]["predicted_structure"],
            reverse=True,
        ),
    }
    filtered = {}
    for name, items in groups.items():
        left, right = {
            "unitable_better_than_flat": ("predicted_structure", "flat"),
            "flat_better_than_unitable": ("flat", "predicted_structure"),
            "gt_better_than_unitable": ("oracle_structure", "predicted_structure"),
        }[name]
        filtered[name] = [
            item for item in items if item["scores"][left] > item["scores"][right]
        ][: args.cases_per_group]
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    args.cases_output.write_text(
        json.dumps({"groups": filtered}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
