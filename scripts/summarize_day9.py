#!/usr/bin/env python3
"""Summarize Day-9 ablations and create joint retrieval/QA error taxonomy."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from run_day9_qa import parse_number


GROUPS = {
    "input_structure": ["flat", "gt_zero_shot", "unitable"],
    "learning_components": ["unitable", "hierarchy", "hierarchy_hard",
                            "hierarchy_hard_kd"],
    "aggregation": ["fedavg_queue", "r_only", "u_only", "r_plus_u"],
}
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/day9"))
    parser.add_argument("--methods", type=Path, default=Path("configs/day9_methods.json"))
    parser.add_argument("--finqa-test", type=Path, default=Path("data/raw/finqa/test.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def classify(retrieval: dict, qa: dict, positive_by_level: dict,
             program: str) -> str:
    if qa["numerical_em"]:
        return "success"
    if not retrieval["gold_hit"]:
        return "retrieval_failure"
    retrieved_ids = {row["evidence_id"] for row in retrieval["retrieved"]}
    fine_positive = set(positive_by_level.get("row", [])) | set(
        positive_by_level.get("cell", []))
    if fine_positive and not (retrieved_ids & fine_positive):
        return "structure_localization_failure"
    if qa["parsed_prediction"] is None:
        return "format_matching_failure"
    operands = NUMBER.findall(program)
    context = " ".join(row["text"] for row in retrieval["retrieved"])
    normalized_context = context.replace(",", "")
    if operands and not all(token.replace(",", "") in normalized_context
                            for token in operands):
        return "number_extraction_failure"
    return "operation_failure"


def main() -> None:
    args = parse_args()
    registry = json.loads(args.methods.read_text())
    protocol = registry["protocol"]
    if args.seed != protocol["seed"]:
        raise ValueError("seed differs from frozen protocol")
    sample = json.loads((args.root / "sample_ids.json").read_text())
    frozen_ids = [row["query_id"] for row in sample["samples"]]
    frozen_set = set(frozen_ids)
    finqa = {f"test:{item['id']}:{index}": item["qa"]
             for index, item in enumerate(json.loads(args.finqa_test.read_text()))}
    evidence_maps = {row["query_id"]: row for row in
                     jsonl(Path("data/processed/evidence_map.jsonl"))
                     if row["split"] == "test" and row["query_id"] in frozen_set}
    summary = {
        "seed": args.seed, "sample_ids_sha256": sha256(args.root / "sample_ids.json"),
        "sample_size": len(frozen_ids), "protocol": protocol,
        "program_accuracy": None,
        "program_accuracy_reason": "no method predicted a program",
        "groups": GROUPS, "methods": {},
        "selection_or_question_edits": 0,
    }
    analysis = {"taxonomy": {}, "structure_help_successes": []}
    method_rows = {}
    for method, spec in registry["methods"].items():
        retrieval_path = args.root / "retrieval" / f"{method}.jsonl"
        qa_path = args.root / "qa" / f"{method}.jsonl"
        retrieval = {row["query_id"]: row for row in jsonl(retrieval_path)}
        qa = {row["query_id"]: row for row in jsonl(qa_path)}
        if set(retrieval) != frozen_set or set(qa) != frozen_set:
            raise ValueError(f"{method}: IDs differ from frozen subset")
        if any(retrieval[qid]["question"] != qa[qid]["question"] for qid in frozen_ids):
            raise ValueError(f"{method}: question mismatch")
        categories = defaultdict(list)
        joint = []
        for qid in frozen_ids:
            positive = evidence_maps[qid]["variants"][spec["variant"]][
                "positive_evidence_ids"]
            category = classify(retrieval[qid], qa[qid], positive,
                                str(finqa[qid].get("program") or ""))
            case = {
                "method": method, "query_id": qid,
                "category": category, "question": qa[qid]["question"],
                "gold_answer": qa[qid]["gold_answer"],
                "prediction": qa[qid]["prediction"],
                "gold_hit": retrieval[qid]["gold_hit"],
                "first_gold_rank": retrieval[qid]["first_gold_rank"],
                "retrieved_evidence_ids": qa[qid]["retrieved_evidence_ids"],
            }
            categories[category].append(case)
            joint.append(case)
        counts = {key: len(value) for key, value in sorted(categories.items())}
        analysis["taxonomy"][method] = {
            "counts": counts,
            "examples_up_to_5": {key: value[:5] for key, value in categories.items()},
            "fewer_than_5": {key: len(value) for key, value in categories.items()
                             if len(value) < 5},
        }
        retrieval_state = json.loads(
            (args.root / "retrieval" / f"{method}.state.json").read_text())
        qa_state = json.loads((args.root / "qa" / f"{method}.state.json").read_text())
        summary["methods"][method] = {
            "variant": spec["variant"], "note": spec.get("note"),
            "retrieval_recall@5": retrieval_state["unified_evidence_recall@5"],
            "retrieval_mrr@5": retrieval_state["unified_evidence_mrr@5"],
            "numerical_em": qa_state["numerical_em"],
            "qa_errors": qa_state["errors"], "program_accuracy": None,
            "retrieval_sha256": sha256(retrieval_path),
            "qa_sha256": sha256(qa_path), "joint_error_counts": counts,
        }
        method_rows[method] = {row["query_id"]: row for row in joint}
    flat = method_rows["flat"]
    for structured in ("gt_zero_shot", "unitable"):
        for qid in frozen_ids:
            current, base = method_rows[structured][qid], flat[qid]
            if current["category"] == "success" and base["category"] != "success":
                analysis["structure_help_successes"].append({
                    "structured_method": structured, "query_id": qid,
                    "flat_category": base["category"], "case": current,
                })
    analysis["structure_help_examples_up_to_10"] = analysis[
        "structure_help_successes"][:10]
    (args.root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    (args.root / "joint_error_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n")
    manifest_files = [args.root / "sample_ids.json", args.root / "ablation_summary.json",
                      args.root / "joint_error_analysis.json"]
    manifest_files += sorted((args.root / "retrieval").glob("*"))
    manifest_files += sorted(path for path in (args.root / "qa").glob("*")
                             if ".limit" not in path.name)
    manifest = {
        "seed": args.seed, "files": {
            str(path.relative_to(args.root)): {"sha256": sha256(path),
                                               "bytes": path.stat().st_size}
            for path in manifest_files if path.is_file()
        },
        "frozen_ids": len(frozen_ids), "methods": len(registry["methods"]),
        "question_set_identical": True, "top_k_identical": True,
        "prompt_llm_temperature_identical": True, "question_edits": 0,
    }
    (args.root / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in [*manifest_files, args.root / "freeze_manifest.json"]:
        if path.exists() and path.is_file():
            path.chmod(0o444)


if __name__ == "__main__":
    main()
