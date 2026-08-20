#!/usr/bin/env python3
"""Dev-only score fusion over frozen Flat/UniTable Top-10 retrieval pools.

No model or embedding is loaded. Candidate evidence IDs are aligned exactly;
the union of the two persisted Top-10 lists is normalized and reranked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any


LOG = logging.getLogger("evaluate_score_fusion")
LEVELS = ("table", "row", "cell")
LEVEL_WEIGHTS = {"table": 0.2, "row": 0.3, "cell": 0.5}
NORMALIZATIONS = ("z-score", "min-max", "rank_score")
ALPHAS = tuple(index / 10 for index in range(11))
GRID_FIELDS = (
    "config_id", "normalization", "alpha", "candidate_alignment",
    "missing_score_policy", "mean_table_pool_size", "mean_row_pool_size",
    "mean_cell_pool_size", "table_ndcg@10", "row_ndcg@10", "cell_ndcg@10",
    "table_recall@5", "row_recall@5", "cell_recall@5", "table_mrr",
    "row_mrr", "cell_mrr", "table_map", "row_map", "cell_map",
    "weighted_ndcg@10", "global_weighted_ndcg@10", "macro_weighted_ndcg@10",
    "worst_client_weighted_ndcg@10", "std_client_weighted_ndcg@10",
    "delta_weighted_ndcg_vs_flat", "delta_row_recall@5_vs_flat",
    "delta_cell_recall@5_vs_flat", "delta_worst_client_vs_flat",
    "passes_numeric_preregistered_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-dev", type=Path,
                        default=Path("outputs/retrieval/zero_shot_flat_dev.json"))
    parser.add_argument("--unitable-dev", type=Path, default=Path(
        "outputs/retrieval/zero_shot_predicted_structure_dev.json"))
    parser.add_argument("--flat-queries", type=Path,
                        default=Path("data/processed/flat/queries_dev.jsonl"))
    parser.add_argument("--unitable-queries", type=Path, default=Path(
        "data/processed/predicted_structure/queries_dev.jsonl"))
    parser.add_argument("--evidence-map", type=Path,
                        default=Path("data/processed/evidence_map.jsonl"))
    parser.add_argument("--partition", type=Path, default=Path(
        "data/processed/federated/company_5clients.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/diagnosis/fusion"))
    parser.add_argument("--log-file", type=Path,
                        default=Path("outputs/logs/diagnosis_fusion.log"))
    parser.add_argument("--default-normalization", choices=NORMALIZATIONS,
                        default="z-score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{number}") from exc
    return rows


def unique(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(candidates: list[dict[str, Any]], method: str) -> tuple[dict[str, float], float]:
    """Return normalized scores and a deterministic below-pool missing score."""
    if not candidates:
        return {}, -1.0
    scores = [float(item["score"]) for item in candidates]
    if method == "min-max":
        low, high = min(scores), max(scores)
        values = ([0.0] * len(scores) if high == low else
                  [(score - low) / (high - low) for score in scores])
        missing = -1.0
    elif method == "z-score":
        center = statistics.fmean(scores)
        scale = statistics.pstdev(scores)
        values = ([0.0] * len(scores) if scale == 0.0 else
                  [(score - center) / scale for score in scores])
        missing = min(values) - 1.0
    elif method == "rank_score":
        count = len(candidates)
        values = ([1.0] if count == 1 else
                  [1.0 - rank / (count - 1) for rank in range(count)])
        missing = -1.0
    else:
        raise ValueError(f"Unknown normalization: {method}")
    return ({str(item["evidence_id"]): value for item, value in zip(candidates, values)},
            missing)


def fuse(flat: list[dict[str, Any]], unitable: list[dict[str, Any]], method: str,
         alpha: float) -> list[tuple[str, float]]:
    flat_scores, flat_missing = normalize(flat, method)
    unitable_scores, unitable_missing = normalize(unitable, method)
    evidence_ids = set(flat_scores) | set(unitable_scores)
    fused = [
        (evidence_id,
         alpha * flat_scores.get(evidence_id, flat_missing)
         + (1.0 - alpha) * unitable_scores.get(evidence_id, unitable_missing))
        for evidence_id in evidence_ids
    ]
    # Exact evidence ID breaks score ties, making every run deterministic.
    return sorted(fused, key=lambda item: (-item[1], item[0]))


def ranking_metrics(ranking: list[tuple[str, float]], positives: set[str],
                    total_positive_count: int) -> dict[str, float]:
    ranks = [rank for rank, (evidence_id, _) in enumerate(ranking, 1)
             if evidence_id in positives]
    denominator = sum(1.0 / math.log2(rank + 1)
                      for rank in range(1, min(total_positive_count, 10) + 1))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ap = (sum(index / rank for index, rank in enumerate(ranks, 1)) /
          total_positive_count if total_positive_count else 0.0)
    return {
        "ndcg@10": dcg / denominator if denominator else 0.0,
        "recall@5": float(any(rank <= 5 for rank in ranks)),
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
        "ap": ap,
    }


def validate(args: argparse.Namespace, flat_report: dict[str, Any],
             unitable_report: dict[str, Any]) -> tuple[
                 dict[str, dict[str, Any]], dict[str, dict[str, Any]],
                 dict[str, dict[str, Any]], dict[str, int]]:
    for label, report in (("Flat", flat_report), ("UniTable", unitable_report)):
        if report.get("protocol", {}).get("split") != "dev":
            raise ValueError(f"{label} result is not dev; refusing test leakage")
    for key in ("model_fingerprint", "pooling", "normalize", "max_length",
                "query_instruction", "ranking"):
        if flat_report["protocol"].get(key) != unitable_report["protocol"].get(key):
            raise ValueError(f"Frozen retrieval protocol mismatch: {key}")
    flat = unique(flat_report["per_query"], "query_id", args.flat_dev)
    unitable = unique(unitable_report["per_query"], "query_id", args.unitable_dev)
    flat_queries = unique(read_jsonl(args.flat_queries), "query_id", args.flat_queries)
    unitable_queries = unique(read_jsonl(args.unitable_queries), "query_id", args.unitable_queries)
    query_ids = set(flat)
    if not (query_ids == set(unitable) == set(flat_queries) == set(unitable_queries)):
        raise ValueError("Flat/UniTable result or query coverage mismatch")
    if len(query_ids) != int(flat_report["query_count"]) or any(
            not query_id.startswith("dev:") for query_id in query_ids):
        raise ValueError("Invalid dev query coverage")
    for query_id in query_ids:
        left, right = flat_queries[query_id], unitable_queries[query_id]
        if (left["question"], left["sample_id"], left["split"]) != (
                right["question"], right["sample_id"], right["split"]):
            raise ValueError(f"Cross-branch query mismatch: {query_id}")

    maps = unique([row for row in read_jsonl(args.evidence_map) if row.get("split") == "dev"],
                  "query_id", args.evidence_map)
    if set(maps) != query_ids:
        raise ValueError("Evidence-map dev coverage mismatch")
    partition = load_json(args.partition)
    owners = {str(query_id): int(client_id)
              for client_id, client in partition["clients"].items()
              for query_id in client["query_ids"]["dev"]}
    if set(owners) != query_ids:
        raise ValueError("Frozen client partition dev coverage mismatch")
    for query_id in query_ids:
        for level in LEVELS:
            left = flat[query_id]["levels"].get(level, {})
            right = unitable[query_id]["levels"].get(level, {})
            if bool(left.get("eligible")) != bool(right.get("eligible")):
                raise ValueError(f"Eligibility mismatch: {query_id}/{level}")
            if left.get("eligible") and (len(left.get("top10", [])) != 10 or
                                         len(right.get("top10", [])) != 10):
                raise ValueError(f"Frozen candidate depth is not 10: {query_id}/{level}")
    return flat, unitable, maps, owners


def evaluate(method: str, alpha: float, flat: dict[str, dict[str, Any]],
             unitable: dict[str, dict[str, Any]], maps: dict[str, dict[str, Any]],
             owners: dict[str, int]) -> dict[str, Any]:
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    pool_sizes: dict[str, list[int]] = {level: [] for level in LEVELS}
    for query_id in flat:
        per_query[query_id] = {}
        mapping = maps[query_id]["variants"]
        for level in LEVELS:
            left = flat[query_id]["levels"].get(level, {})
            if not left.get("eligible"):
                continue
            right = unitable[query_id]["levels"][level]
            ranking = fuse(left["top10"], right["top10"], method, alpha)
            flat_positive = set(mapping["flat"]["positive_evidence_ids"][level])
            unitable_positive = set(
                mapping["predicted_structure"]["positive_evidence_ids"][level])
            positives = flat_positive | unitable_positive
            # Shared table IDs must not inflate the ideal multi-positive count.
            per_query[query_id][level] = ranking_metrics(ranking, positives, len(positives))
            pool_sizes[level].append(len(ranking))
    global_metrics: dict[str, dict[str, float]] = {}
    client_level: dict[int, dict[str, float]] = {cid: {} for cid in sorted(set(owners.values()))}
    for level in LEVELS:
        rows = [levels[level] for levels in per_query.values() if level in levels]
        global_metrics[level] = {
            metric: statistics.fmean(row[metric] for row in rows)
            for metric in ("ndcg@10", "recall@5", "mrr", "ap")
        }
        for client_id in client_level:
            values = [per_query[qid][level]["ndcg@10"] for qid in per_query
                      if owners[qid] == client_id and level in per_query[qid]]
            client_level[client_id][level] = statistics.fmean(values) if values else 0.0
    weighted = sum(LEVEL_WEIGHTS[level] * global_metrics[level]["ndcg@10"]
                   for level in LEVELS)
    clients = [sum(LEVEL_WEIGHTS[level] * values[level] for level in LEVELS)
               for values in client_level.values()]
    row: dict[str, Any] = {
        "normalization": method, "alpha": alpha,
        "candidate_alignment": "exact_evidence_id_top10_union",
        "missing_score_policy": "one_normalized_unit_below_branch_minimum",
        "weighted_ndcg@10": weighted,
        "global_weighted_ndcg@10": weighted,
        "macro_weighted_ndcg@10": statistics.fmean(clients),
        "worst_client_weighted_ndcg@10": min(clients),
        "std_client_weighted_ndcg@10": statistics.pstdev(clients),
    }
    for level in LEVELS:
        row[f"mean_{level}_pool_size"] = statistics.fmean(pool_sizes[level])
        row[f"{level}_ndcg@10"] = global_metrics[level]["ndcg@10"]
        row[f"{level}_recall@5"] = global_metrics[level]["recall@5"]
        row[f"{level}_mrr"] = global_metrics[level]["mrr"]
        row[f"{level}_map"] = global_metrics[level]["ap"]
    return row


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter); stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    LOG.info("Starting Task 5 seed=%d; dev-only frozen Top-10 score fusion", args.seed)
    flat_report, unitable_report = load_json(args.flat_dev), load_json(args.unitable_dev)
    flat, unitable, maps, owners = validate(args, flat_report, unitable_report)
    rows = []
    for method in NORMALIZATIONS:
        for alpha in ALPHAS:
            row = evaluate(method, alpha, flat, unitable, maps, owners)
            row["config_id"] = f"f{len(rows):03d}"
            rows.append(row)
    # The alpha=1 endpoint exactly preserves Flat Top-10 ranking metrics.
    flat_baseline = next(row for row in rows
                         if row["normalization"] == args.default_normalization and
                         row["alpha"] == 1.0)
    for row in rows:
        row["delta_weighted_ndcg_vs_flat"] = (
            row["weighted_ndcg@10"] - flat_baseline["weighted_ndcg@10"])
        row["delta_row_recall@5_vs_flat"] = row["row_recall@5"] - flat_baseline["row_recall@5"]
        row["delta_cell_recall@5_vs_flat"] = row["cell_recall@5"] - flat_baseline["cell_recall@5"]
        row["delta_worst_client_vs_flat"] = (row["worst_client_weighted_ndcg@10"] -
                                               flat_baseline["worst_client_weighted_ndcg@10"])
        row["passes_numeric_preregistered_gate"] = bool(
            row["delta_weighted_ndcg_vs_flat"] >= 0.003 and
            max(row["delta_row_recall@5_vs_flat"], row["delta_cell_recall@5_vs_flat"]) >= 0.01 and
            row["delta_worst_client_vs_flat"] >= -0.005)
    best = max(rows, key=lambda row: (row["weighted_ndcg@10"],
                                      row["worst_client_weighted_ndcg@10"],
                                      row["normalization"] == args.default_normalization,
                                      row["alpha"]))
    passing_rows = [row for row in rows if row["passes_numeric_preregistered_gate"]]
    best_passing = (max(passing_rows, key=lambda row: (
        row["weighted_ndcg@10"], row["worst_client_weighted_ndcg@10"]))
        if passing_rows else None)
    full_corpus_flat_reference = sum(
        LEVEL_WEIGHTS[level] * float(flat_report["metrics"][level]["ndcg@10"])
        for level in LEVELS)
    payload = {
        "schema_version": "1.0", "task": 5, "selection_split": "dev",
        "test_accessed": False, "seed": args.seed, "strategy": "score_fusion",
        "config": {"config_id": best["config_id"],
                   "normalization": best["normalization"], "alpha": best["alpha"],
                   "formula": "alpha*flat + (1-alpha)*unitable",
                   "candidate_alignment": best["candidate_alignment"],
                   "missing_score_policy": best["missing_score_policy"]},
        "metrics": best, "flat_dev_baseline": flat_baseline,
        "full_corpus_flat_reference": {
            "weighted_ndcg@10": full_corpus_flat_reference,
            "source": str(args.flat_dev),
            "comparable_to_fusion_pool": False,
            "reason": "Row/cell variant evidence IDs create a doubled representation pool; use only as an external audit reference.",
        },
        "best_numeric_passing_candidate": best_passing,
        "selection_rule": "maximize dev weighted NDCG@10; tie-break worst-client, default z-score, then alpha",
        "candidate_pool_boundary": (
            "Frozen outputs persist Top-10 only. Fusion reranks the exact evidence-ID union "
            "(<=20); NDCG@10/Recall@5 are reranked-pool metrics and endpoint-auditable. "
            "MRR/MAP are truncated-pool diagnostics, not full-corpus exact metrics."),
        "eligible_for_task6_numeric_gate": bool(passing_rows),
        "note": "Task 5 alone never authorizes formal test; Task 6 must freeze the decision.",
    }
    output_dir = args.output_dir
    grid_path = output_dir / "fusion_dev_grid.csv"
    best_path = output_dir / "fusion_dev_best_config.json"
    summary_path = output_dir / "fusion_dev_summary.md"
    manifest_path = output_dir / "fusion_sha256_manifest.json"
    atomic_csv(grid_path, rows, args.overwrite)
    atomic_text(best_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", args.overwrite)
    passed = len(passing_rows)
    status = ("存在池内端点口径的数值候选；仍须任务6审查口径、稳定性并冻结" if passed else
              "未达到预注册数值门槛；不得据任务5结果运行test")
    passing_text = (f"{best_passing['config_id']}（{best_passing['normalization']}，"
                    f"alpha={best_passing['alpha']:.1f}）" if best_passing else "无")
    summary = f"""# Task 5 冻结 BGE Flat/UniTable 分数后融合（dev-only）

## 协议与边界

- 仅使用883条dev query的冻结Flat/UniTable Top-10候选及分数；未训练、未重新编码、未读取test。
- 按精确evidence_id对齐，候选池为两路Top-10并集；Table平均池大小{best['mean_table_pool_size']:.4f}，Row/Cell均为{best['mean_row_pool_size']:.4f}/{best['mean_cell_pool_size']:.4f}。
- 支持z-score、min-max和rank_score；每种方式搜索alpha=0.0,0.1,...,1.0，共{len(rows)}项。
- 缺失候选分数置于该分支归一化最低分以下1单位。MRR/MAP是截断融合池诊断值，不冒充全语料精确指标。

## 结果

- Flat endpoint weighted NDCG@10：{flat_baseline['weighted_ndcg@10']:.9f}。
- 原冻结全语料Flat参考值：{full_corpus_flat_reference:.9f}；因Row/Cell表示池翻倍，不与融合池绝对值直接比较。
- 最佳配置：{best['config_id']}，normalization={best['normalization']}，alpha={best['alpha']:.1f}。
- 最佳 weighted NDCG@10：{best['weighted_ndcg@10']:.9f}，相对Flat {best['delta_weighted_ndcg_vs_flat']:+.9f}。
- Row Recall@5变化：{best['delta_row_recall@5_vs_flat']:+.9f}；Cell Recall@5变化：{best['delta_cell_recall@5_vs_flat']:+.9f}。
- worst-client weighted NDCG@10变化：{best['delta_worst_client_vs_flat']:+.9f}。
- 通过池内端点数值筛查的配置数：{passed}/{len(rows)}；候选：{passing_text}。判定：{status}。

## 决策边界

本结果只用于dev方案选择。必须由任务6结合任务4、稳定性和预注册条件统一冻结；当前禁止执行正式test。
"""
    atomic_text(summary_path, summary, args.overwrite)
    manifest = {
        "schema_version": "1.0", "selection_split": "dev", "test_accessed": False,
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in (args.flat_dev, args.unitable_dev, args.flat_queries,
                                args.unitable_queries, args.evidence_map, args.partition)],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in (grid_path, best_path, summary_path)],
    }
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    LOG.info("Task 5 complete configs=%d best=%s weighted_ndcg=%.9f passed=%d",
             len(rows), best["config_id"], best["weighted_ndcg@10"], passed)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Task 5 score fusion failed")
        sys.exit(2)
