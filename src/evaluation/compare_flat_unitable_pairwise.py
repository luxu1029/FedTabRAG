#!/usr/bin/env python3
"""Create strict per-query Flat-vs-UniTable retrieval comparison tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any


LOG = logging.getLogger("compare_flat_unitable_pairwise")
LEVELS_FINEST_FIRST = ("cell", "row", "table")
FIELDS = (
    "query_id", "sample_id", "question", "company", "question_type",
    "program_type", "table_size", "evidence_level", "gold_evidence_id",
    "flat_gold_rank", "unitable_gold_rank", "flat_recall@5", "unitable_recall@5",
    "flat_ndcg@10", "unitable_ndcg@10", "flat_mrr", "unitable_mrr",
    "flat_map", "unitable_map", "delta_ndcg", "winner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--flat-dev", type=Path,
                        default=Path("outputs/retrieval/zero_shot_flat_dev.json"))
    parser.add_argument("--unitable-dev", type=Path,
                        default=Path("outputs/retrieval/zero_shot_predicted_structure_dev.json"))
    parser.add_argument("--flat-test", type=Path,
                        default=Path("outputs/retrieval/zero_shot_flat.json"))
    parser.add_argument("--unitable-test", type=Path,
                        default=Path("outputs/retrieval/zero_shot_predicted_structure.json"))
    parser.add_argument("--queries-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--evidence-map", type=Path,
                        default=Path("data/processed/evidence_map.jsonl"))
    parser.add_argument("--finqa-root", type=Path, default=Path("data/raw/finqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnosis"))
    parser.add_argument("--sha-output", type=Path,
                        default=Path("outputs/diagnosis/pairwise_input_sha256.json"))
    parser.add_argument("--log-file", type=Path,
                        default=Path("outputs/logs/diagnosis_pairwise.log"))
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


def unique_by(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def question_type(item: dict[str, Any]) -> str:
    keys = [str(key) for key in (item.get("qa", {}).get("gold_inds") or {})]
    has_table = any(key.startswith("table_") for key in keys)
    has_text = any(key.startswith("text_") for key in keys)
    if has_table and has_text:
        return "hybrid"
    if has_table:
        return "table"
    if has_text:
        return "text"
    return "unknown"


def program_type(item: dict[str, Any]) -> str:
    program = str(item.get("qa", {}).get("program") or "unknown").strip()
    return program.split("(", 1)[0].strip() or "unknown"


def choose_level(mapping: dict[str, Any]) -> str:
    variants = mapping["variants"]
    for level in LEVELS_FINEST_FIRST:
        flat_ids = variants["flat"]["positive_evidence_ids"].get(level, [])
        unitable_ids = variants["predicted_structure"]["positive_evidence_ids"].get(level, [])
        if flat_ids and unitable_ids:
            return level
    raise ValueError(f"No shared eligible evidence level for {mapping['query_id']}")


def level_data(row: dict[str, Any], level: str, query_id: str, variant: str) -> dict[str, Any]:
    value = row.get("levels", {}).get(level)
    if not value or not value.get("eligible"):
        raise ValueError(f"Missing eligible {level} result for {variant} query {query_id}")
    metrics = value.get("metrics", {})
    required = ("recall@5", "ndcg@10", "mrr", "ap")
    missing = [field for field in required if field not in metrics]
    if missing:
        raise ValueError(f"Missing metrics {missing} for {variant} query {query_id}/{level}")
    ranks = [int(rank) for rank in value.get("positive_ranks", [])]
    if not ranks:
        raise ValueError(f"No positive ranks for {variant} query {query_id}/{level}")
    return {"rank": min(ranks), **{field: float(metrics[field]) for field in required}}


def classify(flat: dict[str, Any], unitable: dict[str, Any], tolerance: float) -> str:
    if flat["rank"] > 10 and unitable["rank"] > 10:
        return "both_fail"
    delta = unitable["ndcg@10"] - flat["ndcg@10"]
    if delta > tolerance:
        return "unitable_win"
    if delta < -tolerance:
        return "flat_win"
    if unitable["rank"] < flat["rank"]:
        return "unitable_win"
    if flat["rank"] < unitable["rank"]:
        return "flat_win"
    return "tie"


def atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def split_paths(args: argparse.Namespace, split: str) -> tuple[Path, Path, Path, Path, Path]:
    return (
        getattr(args, f"flat_{split}"), getattr(args, f"unitable_{split}"),
        args.queries_root / "flat" / f"queries_{split}.jsonl",
        args.queries_root / "predicted_structure" / f"queries_{split}.jsonl",
        args.finqa_root / f"{split}.json",
    )


def build_split(args: argparse.Namespace, split: str, maps: dict[str, dict[str, Any]]) -> tuple[list[dict], list[Path]]:
    flat_path, unitable_path, queries_path, unitable_queries_path, finqa_path = split_paths(
        args, split
    )
    flat_report, unitable_report = load_json(flat_path), load_json(unitable_path)
    if flat_report.get("variant") != "flat":
        raise ValueError(f"Expected flat variant in {flat_path}")
    if unitable_report.get("variant") != "predicted_structure":
        raise ValueError(f"Expected predicted_structure variant in {unitable_path}")
    protocol_fields = ("model_fingerprint", "pooling", "normalize", "max_length",
                       "query_instruction", "ranking", "map_definition")
    flat_protocol, unitable_protocol = flat_report.get("protocol", {}), unitable_report.get(
        "protocol", {}
    )
    mismatched_protocol = [field for field in protocol_fields
                           if flat_protocol.get(field) != unitable_protocol.get(field)]
    if mismatched_protocol:
        raise ValueError(f"Retrieval protocol mismatch for {split}: {mismatched_protocol}")
    flat = unique_by(flat_report.get("per_query", []), "query_id", flat_path)
    unitable = unique_by(unitable_report.get("per_query", []), "query_id", unitable_path)
    queries = unique_by(read_jsonl(queries_path), "query_id", queries_path)
    unitable_queries = unique_by(
        read_jsonl(unitable_queries_path), "query_id", unitable_queries_path
    )
    if set(queries) != set(unitable_queries):
        raise ValueError(f"Flat/UniTable query ID mismatch for {split}")
    for query_id, query in queries.items():
        other = unitable_queries[query_id]
        for field in ("sample_id", "sample_key", "question", "company", "split"):
            if query.get(field) != other.get(field):
                raise ValueError(f"Flat/UniTable query field mismatch {query_id}/{field}")
    finqa = load_json(finqa_path)
    raw = {f"{split}:{item['id']}:{index}": item for index, item in enumerate(finqa)}
    ids = set(queries)
    for name, values in (("flat", set(flat)), ("unitable", set(unitable)),
                         ("evidence_map", set(maps)), ("FinQA", set(raw))):
        missing, extra = ids - values, values - ids if name in {"flat", "unitable"} else set()
        if missing or extra:
            raise ValueError(
                f"{split} {name} ID mismatch: missing={len(missing)} extra={len(extra)} "
                f"first_missing={next(iter(sorted(missing)), None)}"
            )
    rows = []
    for query_id in sorted(ids):
        query, source, mapping = queries[query_id], raw[query_id], maps[query_id]
        for variant_name, report_row in (("flat", flat[query_id]),
                                         ("unitable", unitable[query_id])):
            if report_row.get("sample_id") != query.get("sample_id") or \
                    report_row.get("question") != query.get("question"):
                raise ValueError(f"{variant_name} retrieval/query metadata mismatch for {query_id}")
        if mapping.get("split") != split:
            raise ValueError(f"Evidence-map split mismatch for {query_id}")
        level = choose_level(mapping)
        flat_value = level_data(flat[query_id], level, query_id, "flat")
        unitable_value = level_data(unitable[query_id], level, query_id, "unitable")
        flat_ids = sorted(mapping["variants"]["flat"]["positive_evidence_ids"][level])
        unitable_ids = sorted(
            mapping["variants"]["predicted_structure"]["positive_evidence_ids"][level]
        )
        delta = unitable_value["ndcg@10"] - flat_value["ndcg@10"]
        rows.append({
            "query_id": query_id, "sample_id": query["sample_id"],
            "question": query["question"], "company": query.get("company", "NA"),
            "question_type": question_type(source), "program_type": program_type(source),
            "table_size": sum(len(table_row) for table_row in source.get("table", [])),
            "evidence_level": level,
            "gold_evidence_id": json.dumps(
                {"flat": flat_ids, "unitable": unitable_ids}, ensure_ascii=False,
                separators=(",", ":")),
            "flat_gold_rank": flat_value["rank"],
            "unitable_gold_rank": unitable_value["rank"],
            "flat_recall@5": flat_value["recall@5"],
            "unitable_recall@5": unitable_value["recall@5"],
            "flat_ndcg@10": flat_value["ndcg@10"],
            "unitable_ndcg@10": unitable_value["ndcg@10"],
            "flat_mrr": flat_value["mrr"], "unitable_mrr": unitable_value["mrr"],
            "flat_map": flat_value["ap"], "unitable_map": unitable_value["ap"],
            "delta_ndcg": delta,
            "winner": classify(flat_value, unitable_value, args.tie_tolerance),
        })
    return rows, [flat_path, unitable_path, queries_path, unitable_queries_path, finqa_path]


def main() -> int:
    args = parse_args()
    if args.tie_tolerance < 0 or not math.isfinite(args.tie_tolerance):
        raise ValueError("--tie-tolerance must be finite and non-negative")
    configure_logging(args.log_file)
    LOG.info("Starting split=%s seed=%d (deterministic comparison; no training)",
             args.split, args.seed)
    maps = unique_by(read_jsonl(args.evidence_map), "query_id", args.evidence_map)
    splits = ("dev", "test") if args.split == "all" else (args.split,)
    inputs = {args.evidence_map}
    outputs = []
    for split in splits:
        rows, split_inputs = build_split(args, split, maps)
        output = args.output_dir / f"pairwise_flat_unitable_{split}.csv"
        atomic_csv(output, rows, args.overwrite)
        inputs.update(split_inputs)
        outputs.append(output)
        counts = {name: sum(row["winner"] == name for row in rows)
                  for name in ("unitable_win", "flat_win", "tie", "both_fail")}
        LOG.info("Wrote split=%s rows=%d winners=%s path=%s", split, len(rows), counts, output)
    sha_payload = {
        "schema_version": "1.0", "seed": args.seed, "split": args.split,
        "inputs": [{"path": str(path), "bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in sorted(inputs)],
        "outputs": [{"path": str(path), "rows": sum(1 for _ in path.open()) - 1,
                     "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in outputs],
        "winner_definition": {
            "both_fail": "both best positive ranks are > 10",
            "win": "larger NDCG@10; on equal NDCG, lower best-positive rank",
            "tie": "NDCG@10 within tolerance and equal best-positive rank",
            "tie_tolerance": args.tie_tolerance,
        },
        "evidence_level_definition": "finest level eligible in both variants: cell > row > table",
        "gold_evidence_id_encoding": "JSON object with flat and unitable positive-ID arrays",
    }
    if args.sha_output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.sha_output}; use --overwrite explicitly")
    args.sha_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.sha_output.with_suffix(args.sha_output.suffix + ".tmp")
    temporary.write_text(json.dumps(sha_payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(args.sha_output)
    LOG.info("Wrote SHA256 manifest %s", args.sha_output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Pairwise comparison failed")
        sys.exit(2)
