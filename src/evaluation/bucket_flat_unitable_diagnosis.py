#!/usr/bin/env python3
"""Merge UniTable quality metadata and bucket Flat-vs-UniTable paired retrieval results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


LOG = logging.getLogger("bucket_flat_unitable_diagnosis")
METRIC_FIELDS = (
    "flat_ndcg@10", "unitable_ndcg@10", "delta_ndcg",
    "flat_recall@5", "unitable_recall@5",
)
BUCKET_FIELDS = (
    "split", "bucket_dimension", "bucket", "count",
    "flat_ndcg@10", "unitable_ndcg@10", "delta_ndcg",
    "flat_recall@5", "unitable_recall@5",
    "unitable_win_rate", "flat_win_rate", "tie_rate", "both_fail_rate",
    "mean_s_teds", "valid_html_rate", "cell_count_exact_rate",
    "mean_cell_count_ratio", "mean_predicted_cells", "mean_true_cells",
    "mean_predicted_rows", "mean_true_rows", "mean_predicted_cols", "mean_true_cols",
)
ENRICHED_FIELDS = (
    "query_id", "sample_id", "question", "company", "question_type", "program_type",
    "table_size", "table_size_bucket", "evidence_level", "gold_evidence_id",
    "s_teds", "s_teds_bucket", "valid_html", "cell_count_exact",
    "cell_count_consistency_bucket", "cell_count_ratio", "confidence", "status",
    "predicted_cells", "true_cells", "predicted_rows", "true_rows",
    "predicted_cols", "true_cols", "row_absolute_error", "col_absolute_error",
    "flat_gold_rank", "unitable_gold_rank", "flat_recall@5", "unitable_recall@5",
    "flat_ndcg@10", "unitable_ndcg@10", "flat_mrr", "unitable_mrr",
    "flat_map", "unitable_map", "delta_ndcg", "winner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-dev", type=Path,
                        default=Path("outputs/diagnosis/pairwise_flat_unitable_dev.csv"))
    parser.add_argument("--pairwise-test", type=Path,
                        default=Path("outputs/diagnosis/pairwise_flat_unitable_test.csv"))
    parser.add_argument("--structure-metrics", type=Path,
                        default=Path("outputs/unitable/structure_metrics_per_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/diagnosis/buckets"))
    parser.add_argument("--summary", type=Path,
                        default=Path("outputs/diagnosis/diagnosis_summary.md"))
    parser.add_argument("--sha-output", type=Path,
                        default=Path("outputs/diagnosis/bucket_input_sha256.json"))
    parser.add_argument("--log-file", type=Path,
                        default=Path("outputs/logs/diagnosis_buckets.log"))
    parser.add_argument("--small-max-cells", type=int, default=20)
    parser.add_argument("--medium-max-cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate quantile of empty values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def steds_bucket(score: float, boundaries: tuple[float, float, float]) -> str:
    if score <= boundaries[0]:
        return "q1_low"
    if score <= boundaries[1]:
        return "q2"
    if score <= boundaries[2]:
        return "q3"
    return "q4_high"


def table_size_bucket(cells: int, small_max: int, medium_max: int) -> str:
    if cells <= small_max:
        return "small"
    if cells <= medium_max:
        return "medium"
    return "large"


def atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]],
               overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def unique_by(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def enrich(pairwise: list[dict[str, str]], structure: dict[str, dict[str, Any]],
           split: str, boundaries: tuple[float, float, float], args: argparse.Namespace
           ) -> list[dict[str, Any]]:
    ids = {row["query_id"] for row in pairwise}
    structure_ids = {key for key, value in structure.items() if value.get("split") == split}
    if ids != structure_ids:
        raise ValueError(
            f"{split} pairwise/structure ID mismatch: missing={len(ids-structure_ids)} "
            f"extra={len(structure_ids-ids)}"
        )
    result = []
    for row in pairwise:
        metric = structure[row["query_id"]]
        cells = int(row["table_size"])
        score = float(metric["s_teds"])
        merged = dict(row)
        merged.update({
            "table_size": cells,
            "table_size_bucket": table_size_bucket(
                cells, args.small_max_cells, args.medium_max_cells),
            "s_teds": score,
            "s_teds_bucket": steds_bucket(score, boundaries),
            "valid_html": bool(metric["valid_html"]),
            "cell_count_exact": bool(metric["cell_count_exact"]),
            "cell_count_consistency_bucket": (
                "exact" if metric["cell_count_exact"] else "mismatch"),
            "cell_count_ratio": float(metric["cell_count_ratio"]),
            "confidence": float(metric["confidence"]), "status": metric["status"],
            "predicted_cells": int(metric["predicted_cells"]),
            "true_cells": int(metric["true_cells"]),
            "predicted_rows": int(metric["predicted_rows"]),
            "true_rows": int(metric["true_rows"]),
            "predicted_cols": int(metric["predicted_cols"]),
            "true_cols": int(metric["true_cols"]),
            "row_absolute_error": int(metric["row_absolute_error"]),
            "col_absolute_error": int(metric["col_absolute_error"]),
        })
        for field in METRIC_FIELDS:
            merged[field] = float(row[field])
        result.append(merged)
    return result


def aggregate(split: str, dimension: str, rows: list[dict[str, Any]],
              key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    output = []
    for name in sorted(groups):
        values = groups[name]
        count = len(values)
        output.append({
            "split": split, "bucket_dimension": dimension, "bucket": name, "count": count,
            **{field: statistics.fmean(float(row[field]) for row in values)
               for field in METRIC_FIELDS},
            "unitable_win_rate": sum(row["winner"] == "unitable_win" for row in values) / count,
            "flat_win_rate": sum(row["winner"] == "flat_win" for row in values) / count,
            "tie_rate": sum(row["winner"] == "tie" for row in values) / count,
            "both_fail_rate": sum(row["winner"] == "both_fail" for row in values) / count,
            "mean_s_teds": statistics.fmean(row["s_teds"] for row in values),
            "valid_html_rate": sum(row["valid_html"] for row in values) / count,
            "cell_count_exact_rate": sum(row["cell_count_exact"] for row in values) / count,
            "mean_cell_count_ratio": statistics.fmean(row["cell_count_ratio"] for row in values),
            **{f"mean_{field}": statistics.fmean(row[field] for row in values)
               for field in ("predicted_cells", "true_cells", "predicted_rows", "true_rows",
                             "predicted_cols", "true_cols")},
        })
    if sum(row["count"] for row in output) != len(rows):
        raise AssertionError(f"Bucket coverage failure for {split}/{dimension}")
    for field in METRIC_FIELDS:
        weighted = sum(row[field] * row["count"] for row in output) / len(rows)
        direct = statistics.fmean(row[field] for row in rows)
        if abs(weighted - direct) > 1e-12:
            raise AssertionError(f"Weighted aggregation failure {split}/{dimension}/{field}")
    return output


def find_bucket(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in rows if row["bucket"] == name)


def make_summary(boundaries: tuple[float, float, float], results: dict[tuple[str, str], list[dict]],
                 enriched: dict[str, list[dict]], args: argparse.Namespace) -> str:
    dev = enriched["dev"]
    test = enriched["test"]
    overall_dev = statistics.fmean(row["delta_ndcg"] for row in dev)
    overall_test = statistics.fmean(row["delta_ndcg"] for row in test)
    steds_dev = results[("dev", "steds")]
    cell_dev = results[("dev", "cell_count")]
    high = find_bucket(steds_dev, "q4_high")
    low = find_bucket(steds_dev, "q1_low")
    exact = find_bucket(cell_dev, "exact")
    mismatch = next((row for row in cell_dev if row["bucket"] == "mismatch"), None)
    high_interpretation = (
        "高S-TEDS桶仍低于Flat，支持优先检查结构序列化/编码问题。"
        if high["delta_ndcg"] < 0 else
        "高S-TEDS桶出现非负增益，说明结构可能存在局部贡献，可在后续dev门控中验证。"
    )
    if mismatch is None:
        mismatch_text = "dev中没有cell-count mismatch样本，不能据此判断门控价值。"
    else:
        mismatch_text = (
            f"cell-count mismatch桶delta={mismatch['delta_ndcg']:.6f}，"
            f"exact桶delta={exact['delta_ndcg']:.6f}。"
            + ("错位样本退化更明显，可作为后续门控诊断特征。"
               if mismatch["delta_ndcg"] < exact["delta_ndcg"] else
               "错位桶未表现出更强退化，不能仅凭cell-count设计门控。")
        )
    lines = [
        "# Flat vs UniTable 分桶诊断摘要（任务2）", "",
        "## 协议边界", "",
        "- S-TEDS四分位边界只由dev计算，并原样应用到test。",
        f"- dev四分位边界：Q25={boundaries[0]:.6f}，Q50={boundaries[1]:.6f}，"
        f"Q75={boundaries[2]:.6f}。",
        f"- 表格大小边界预注册为small≤{args.small_max_cells}、"
        f"medium≤{args.medium_max_cells}、large>{args.medium_max_cells}个单元格。",
        "- test只用于描述性解释；本脚本不搜索、不选择、不冻结门控阈值或融合权重。",
        "", "## 总体配对结果", "",
        f"- dev：n={len(dev)}，mean delta_ndcg={overall_dev:.6f}。",
        f"- test：n={len(test)}，mean delta_ndcg={overall_test:.6f}（仅解释）。",
        "", "## 关键dev诊断", "",
        f"- q1_low：n={low['count']}，delta_ndcg={low['delta_ndcg']:.6f}，"
        f"UniTable win rate={low['unitable_win_rate']:.4f}。",
        f"- q4_high：n={high['count']}，delta_ndcg={high['delta_ndcg']:.6f}，"
        f"UniTable win rate={high['unitable_win_rate']:.4f}。",
        f"- {high_interpretation}",
        f"- {mismatch_text}",
        "", "## 结论边界", "",
        "本任务只定位差异来源，不据此选择阈值，也不允许test影响任务4-6的方案选择。"
        "门控是否有效必须由后续冻结BGE的dev网格实验单独验证。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.small_max_cells < 1 or args.medium_max_cells <= args.small_max_cells:
        raise ValueError("Table-size thresholds must satisfy 1 <= small < medium")
    configure_logging(args.log_file)
    LOG.info("Starting Task-2 diagnosis seed=%d; dev defines buckets, test is explanatory only",
             args.seed)
    pairwise = {"dev": read_csv(args.pairwise_dev), "test": read_csv(args.pairwise_test)}
    for split, rows in pairwise.items():
        expected_prefix = f"{split}:"
        if not rows or len({row["query_id"] for row in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate pairwise rows for {split}")
        if any(not row["query_id"].startswith(expected_prefix) for row in rows):
            raise ValueError(f"Split leakage in {split} pairwise input")
    structure = unique_by(read_jsonl(args.structure_metrics), "sample_key", args.structure_metrics)
    dev_scores = [float(structure[row["query_id"]]["s_teds"]) for row in pairwise["dev"]]
    boundaries = tuple(quantile(dev_scores, value) for value in (0.25, 0.5, 0.75))
    enriched = {split: enrich(rows, structure, split, boundaries, args)
                for split, rows in pairwise.items()}
    dimensions: dict[str, Callable[[dict[str, Any]], str]] = {
        "steds": lambda row: row["s_teds_bucket"],
        "cell_count": lambda row: row["cell_count_consistency_bucket"],
        "table_size": lambda row: row["table_size_bucket"],
        "question_type": lambda row: row["question_type"],
        "program_type": lambda row: row["program_type"],
        "evidence_level": lambda row: row["evidence_level"],
    }
    outputs = []
    results = {}
    for split in ("dev", "test"):
        enriched_path = args.output_dir / f"pairwise_with_structure_{split}.csv"
        atomic_csv(enriched_path, ENRICHED_FIELDS, enriched[split], args.overwrite)
        outputs.append(enriched_path)
        for dimension, key in dimensions.items():
            rows = aggregate(split, dimension, enriched[split], key)
            results[(split, dimension)] = rows
            path = args.output_dir / f"bucket_by_{dimension}_{split}.csv"
            atomic_csv(path, BUCKET_FIELDS, rows, args.overwrite)
            outputs.append(path)
            LOG.info("Wrote split=%s dimension=%s buckets=%d", split, dimension, len(rows))
    summary_text = make_summary(boundaries, results, enriched, args)
    atomic_text(args.summary, summary_text, args.overwrite)
    outputs.append(args.summary)
    manifest = {
        "schema_version": "1.0", "seed": args.seed,
        "selection_policy": "dev defines quartiles; test explanatory only; no threshold selection",
        "dev_steds_quartile_boundaries": list(boundaries),
        "table_size_boundaries": {"small_max_cells": args.small_max_cells,
                                  "medium_max_cells": args.medium_max_cells},
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in (args.pairwise_dev, args.pairwise_test, args.structure_metrics)],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in outputs],
    }
    if args.sha_output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.sha_output}; use --overwrite")
    atomic_text(args.sha_output, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    LOG.info("Task-2 complete outputs=%d manifest=%s", len(outputs), args.sha_output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Task-2 bucket diagnosis failed")
        sys.exit(2)
