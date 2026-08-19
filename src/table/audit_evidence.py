#!/usr/bin/env python3
"""Audit Day-3 corpora for text multiset consistency, leakage, and references."""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


LOG = logging.getLogger("audit_evidence")
VARIANTS = ("flat", "oracle_structure", "predicted_structure")
FORBIDDEN = ("program_re", "exe_ans", "gold_inds", "\"answer\":", "\"program\":")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspection-output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    corpora = {variant: read_jsonl(args.corpus_root / variant / "corpus.jsonl") for variant in VARIANTS}
    ids = {}
    duplicates = {}
    leakage = {}
    cell_multisets = {}
    for variant, items in corpora.items():
        evidence_ids = [item["evidence_id"] for item in items]
        ids[variant] = set(evidence_ids)
        duplicates[variant] = len(evidence_ids) - len(ids[variant])
        leakage[variant] = [
            item["evidence_id"]
            for item in items
            if any(token in item["text"].lower() for token in FORBIDDEN)
        ]
        grouped = defaultdict(list)
        for item in items:
            if item["level"] == "cell" and item.get("cell_text"):
                grouped[item["sample_key"]].append(item["cell_text"])
        cell_multisets[variant] = {key: Counter(values) for key, values in grouped.items()}

    all_keys = set().union(*(set(values) for values in cell_multisets.values()))
    multiset_mismatches = []
    for key in sorted(all_keys):
        baseline = cell_multisets["flat"].get(key, Counter())
        for variant in VARIANTS[1:]:
            if cell_multisets[variant].get(key, Counter()) != baseline:
                multiset_mismatches.append({"sample_key": key, "variant": variant})

    maps = read_jsonl(args.evidence_map)
    bad_references = []
    for entry in maps:
        for variant, mapping in entry["variants"].items():
            for level, evidence_ids in mapping["positive_evidence_ids"].items():
                for evidence_id in evidence_ids:
                    if evidence_id not in ids[variant]:
                        bad_references.append(
                            {"query_id": entry["query_id"], "variant": variant, "level": level, "evidence_id": evidence_id}
                        )

    rng = random.Random(args.seed)
    selected = rng.sample(maps, min(args.sample_size, len(maps)))
    inspection = []
    for entry in selected:
        inspection.append(
            {
                "query_id": entry["query_id"],
                "sample_id": entry["sample_id"],
                "split": entry["split"],
                "variants": {
                    variant: {
                        "fully_mapped": value["fully_mapped"],
                        "mapping_details": value["mapping_details"],
                    }
                    for variant, value in entry["variants"].items()
                },
            }
        )
    atomic_json(args.inspection_output, {"seed": args.seed, "count": len(inspection), "samples": inspection})
    report = {
        "corpus_counts": {variant: len(items) for variant, items in corpora.items()},
        "duplicate_evidence_ids": duplicates,
        "leakage_counts": {variant: len(values) for variant, values in leakage.items()},
        "leakage_examples": {variant: values[:20] for variant, values in leakage.items()},
        "cell_multiset_mismatch_count": len(multiset_mismatches),
        "cell_multiset_mismatches": multiset_mismatches[:100],
        "bad_reference_count": len(bad_references),
        "bad_references": bad_references[:100],
        "inspection_count": len(inspection),
        "passed": (
            not any(duplicates.values())
            and not any(leakage.values())
            and not multiset_mismatches
            and not bad_references
        ),
    }
    atomic_json(args.output, report)
    LOG.info("Audit passed=%s report=%s", report["passed"], args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
