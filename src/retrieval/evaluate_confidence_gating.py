#!/usr/bin/env python3
"""Dev-only confidence gating over frozen Flat and UniTable retrieval results.

This script never encodes text or loads a model.  It selects, per query, one of
two already-frozen retrieval result records and aggregates their stored exact
ranking metrics globally and over the frozen five-client partition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger("evaluate_confidence_gating")
LEVELS = ("table", "row", "cell")
LEVEL_WEIGHTS = {"table": 0.2, "row": 0.3, "cell": 0.5}
METRICS = ("ndcg@10", "recall@5", "mrr", "ap")
COMPOSITE_WEIGHTS = {
    "s_teds": 0.40,
    "cell_count_consistency": 0.25,
    "valid_html": 0.15,
    "confidence": 0.20,
}
GRID_FIELDS = (
    "config_id", "gate_family", "s_teds_threshold",
    "cell_count_consistency_threshold", "require_valid_html",
    "confidence_threshold", "composite_r_threshold", "unitable_queries",
    "unitable_fraction", "table_ndcg@10", "row_ndcg@10", "cell_ndcg@10",
    "table_recall@5", "row_recall@5", "cell_recall@5", "table_mrr",
    "row_mrr", "cell_mrr", "table_map", "row_map", "cell_map",
    "weighted_ndcg@10", "macro_weighted_ndcg@10",
    "worst_client_weighted_ndcg@10", "std_client_weighted_ndcg@10",
    "delta_weighted_ndcg_vs_flat", "delta_row_recall@5_vs_flat",
    "delta_cell_recall@5_vs_flat", "delta_worst_client_vs_flat",
    "passes_preregistered_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-dev", type=Path,
                        default=Path("outputs/retrieval/zero_shot_flat_dev.json"))
    parser.add_argument("--unitable-dev", type=Path, default=Path(
        "outputs/retrieval/zero_shot_predicted_structure_dev.json"))
    parser.add_argument("--pairwise-dev", type=Path, default=Path(
        "outputs/diagnosis/pairwise_flat_unitable_dev.csv"))
    parser.add_argument("--structure-metrics", type=Path, default=Path(
        "outputs/unitable/structure_metrics_per_sample.jsonl"))
    parser.add_argument("--evidence-map", type=Path,
                        default=Path("data/processed/evidence_map.jsonl"))
    parser.add_argument("--partition", type=Path, default=Path(
        "data/processed/federated/company_5clients.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/diagnosis/gating"))
    parser.add_argument("--log-file", type=Path,
                        default=Path("outputs/logs/diagnosis_gating.log"))
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(rows: Iterable[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean(rows: list[dict[str, float]], metric: str) -> float:
    return statistics.fmean(row[metric] for row in rows) if rows else 0.0


def normalized_consistency(row: dict[str, Any]) -> float:
    """Continuous cell-count agreement in [0, 1], exact matches equal one."""
    if bool(row.get("cell_count_exact")):
        return 1.0
    predicted, true = int(row["predicted_cells"]), int(row["true_cells"])
    if max(predicted, true) == 0:
        return 1.0
    return min(predicted, true) / max(predicted, true)


def reliability(row: dict[str, Any]) -> float:
    parts = {
        "s_teds": min(1.0, max(0.0, float(row["s_teds"]))),
        "cell_count_consistency": normalized_consistency(row),
        "valid_html": float(bool(row["valid_html"])),
        "confidence": min(1.0, max(0.0, float(row["confidence"]))),
    }
    return sum(COMPOSITE_WEIGHTS[key] * value for key, value in parts.items())


def use_unitable(structure: dict[str, Any], config: dict[str, Any]) -> bool:
    if config.get("force") == "flat":
        return False
    if config.get("force") == "unitable":
        return True
    checks = []
    if config.get("s_teds_threshold") is not None:
        checks.append(float(structure["s_teds"]) >= config["s_teds_threshold"])
    if config.get("cell_count_consistency_threshold") is not None:
        checks.append(normalized_consistency(structure) >=
                      config["cell_count_consistency_threshold"])
    if config.get("require_valid_html"):
        checks.append(bool(structure["valid_html"]))
    if config.get("confidence_threshold") is not None:
        checks.append(float(structure["confidence"]) >= config["confidence_threshold"])
    if config.get("composite_r_threshold") is not None:
        checks.append(reliability(structure) >= config["composite_r_threshold"])
    return bool(checks) and all(checks)


def make_grid(structure: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(structure.values())
    quantiles = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    steds = sorted({quantile([float(x["s_teds"]) for x in values], q) for q in quantiles})
    consistency = sorted({quantile([normalized_consistency(x) for x in values], q)
                          for q in quantiles})
    confidence = sorted({quantile([float(x["confidence"]) for x in values], q)
                         for q in quantiles})
    composite = sorted({quantile([reliability(x) for x in values], q) for q in quantiles})
    configs: list[dict[str, Any]] = [
        {"gate_family": "baseline_flat", "force": "flat"},
        {"gate_family": "baseline_unitable", "force": "unitable"},
    ]
    configs += [{"gate_family": "s_teds", "s_teds_threshold": x} for x in steds]
    configs += [{"gate_family": "cell_count_consistency",
                 "cell_count_consistency_threshold": x} for x in consistency]
    configs += [{"gate_family": "valid_html", "require_valid_html": True}]
    configs += [{"gate_family": "confidence", "confidence_threshold": x}
                for x in confidence]
    configs += [{"gate_family": "composite_r", "composite_r_threshold": x}
                for x in composite]
    # Joint quality gates test the explicitly requested filters without an
    # unmanageably large Cartesian product.
    configs += [
        {"gate_family": "joint_quality", "s_teds_threshold": s,
         "cell_count_consistency_threshold": c, "require_valid_html": True}
        for s in steds for c in consistency
    ]
    for index, config in enumerate(configs):
        config["config_id"] = f"g{index:03d}"
    return configs


def validate_inputs(args: argparse.Namespace, flat: dict[str, Any],
                    unitable: dict[str, Any]) -> tuple[dict[str, dict[str, Any]],
                                                       dict[str, dict[str, Any]], dict[str, int]]:
    for name, report in (("Flat", flat), ("UniTable", unitable)):
        if report.get("protocol", {}).get("split") != "dev":
            raise ValueError(f"{name} input is not dev; refusing possible test leakage")
    protocol_keys = ("model_fingerprint", "pooling", "normalize", "max_length",
                     "query_instruction", "ranking")
    for key in protocol_keys:
        if flat["protocol"].get(key) != unitable["protocol"].get(key):
            raise ValueError(f"Frozen retrieval protocol mismatch: {key}")
    flat_rows = unique(flat["per_query"], "query_id", args.flat_dev)
    unitable_rows = unique(unitable["per_query"], "query_id", args.unitable_dev)
    if flat_rows.keys() != unitable_rows.keys() or len(flat_rows) != int(flat["query_count"]):
        raise ValueError("Flat/UniTable query coverage mismatch")
    if any(not query_id.startswith("dev:") for query_id in flat_rows):
        raise ValueError("Non-dev query found in frozen retrieval results")

    with args.pairwise_dev.open(newline="", encoding="utf-8") as handle:
        pairwise = unique(csv.DictReader(handle), "query_id", args.pairwise_dev)
    if pairwise.keys() != flat_rows.keys():
        raise ValueError("Pairwise dev coverage differs from retrieval results")
    evidence = {str(row["query_id"]) for row in read_jsonl(args.evidence_map)
                if row.get("split") == "dev"}
    if evidence != set(flat_rows):
        raise ValueError("Evidence-map dev coverage differs from retrieval results")

    partition = load_json(args.partition)
    owners = {str(query_id): int(client_id)
              for client_id, client in partition["clients"].items()
              for query_id in client["query_ids"]["dev"]}
    if owners.keys() != flat_rows.keys():
        raise ValueError("Frozen client partition dev coverage mismatch")
    return flat_rows, unitable_rows, owners


def evaluate(config: dict[str, Any], flat: dict[str, dict[str, Any]],
             unitable: dict[str, dict[str, Any]], structure: dict[str, dict[str, Any]],
             owners: dict[str, int]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    chosen = 0
    for query_id in flat:
        choose_u = use_unitable(structure[query_id], config)
        selected[query_id] = unitable[query_id] if choose_u else flat[query_id]
        chosen += int(choose_u)
    level_values: dict[str, dict[str, float]] = {}
    client_level: dict[int, dict[str, float]] = {cid: {} for cid in sorted(set(owners.values()))}
    for level in LEVELS:
        eligible = [row["levels"][level]["metrics"] for row in selected.values()
                    if row["levels"].get(level, {}).get("eligible")]
        level_values[level] = {metric: mean(eligible, metric) for metric in METRICS}
        for client_id in client_level:
            rows = [selected[qid]["levels"][level]["metrics"] for qid in selected
                    if owners[qid] == client_id and
                    selected[qid]["levels"].get(level, {}).get("eligible")]
            client_level[client_id][level] = mean(rows, "ndcg@10")
    weighted = sum(LEVEL_WEIGHTS[level] * level_values[level]["ndcg@10"] for level in LEVELS)
    per_client = [sum(LEVEL_WEIGHTS[level] * values[level] for level in LEVELS)
                  for values in client_level.values()]
    row: dict[str, Any] = {
        "config_id": config["config_id"], "gate_family": config["gate_family"],
        "s_teds_threshold": config.get("s_teds_threshold"),
        "cell_count_consistency_threshold": config.get(
            "cell_count_consistency_threshold"),
        "require_valid_html": bool(config.get("require_valid_html", False)),
        "confidence_threshold": config.get("confidence_threshold"),
        "composite_r_threshold": config.get("composite_r_threshold"),
        "unitable_queries": chosen, "unitable_fraction": chosen / len(selected),
        "weighted_ndcg@10": weighted,
        "macro_weighted_ndcg@10": statistics.fmean(per_client),
        "worst_client_weighted_ndcg@10": min(per_client),
        "std_client_weighted_ndcg@10": statistics.pstdev(per_client),
    }
    for level in LEVELS:
        row[f"{level}_ndcg@10"] = level_values[level]["ndcg@10"]
        row[f"{level}_recall@5"] = level_values[level]["recall@5"]
        row[f"{level}_mrr"] = level_values[level]["mrr"]
        row[f"{level}_map"] = level_values[level]["ap"]
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
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter); stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    LOG.info("Starting Task 4 seed=%d; dev-only frozen-result gating", args.seed)

    flat_report, unitable_report = load_json(args.flat_dev), load_json(args.unitable_dev)
    flat, unitable, owners = validate_inputs(args, flat_report, unitable_report)
    all_structure = unique(read_jsonl(args.structure_metrics), "sample_key", args.structure_metrics)
    structure = {qid: all_structure[qid] for qid in flat if qid in all_structure}
    if structure.keys() != flat.keys():
        raise ValueError("Structure metrics do not cover every dev query")
    configs = make_grid(structure)
    rows = [evaluate(config, flat, unitable, structure, owners) for config in configs]
    baseline = next(row for row in rows if row["gate_family"] == "baseline_flat")
    for row in rows:
        row["delta_weighted_ndcg_vs_flat"] = (
            row["weighted_ndcg@10"] - baseline["weighted_ndcg@10"])
        row["delta_row_recall@5_vs_flat"] = row["row_recall@5"] - baseline["row_recall@5"]
        row["delta_cell_recall@5_vs_flat"] = row["cell_recall@5"] - baseline["cell_recall@5"]
        row["delta_worst_client_vs_flat"] = (row["worst_client_weighted_ndcg@10"] -
                                               baseline["worst_client_weighted_ndcg@10"])
        row["passes_preregistered_gate"] = bool(
            row["delta_weighted_ndcg_vs_flat"] >= 0.003 and
            max(row["delta_row_recall@5_vs_flat"], row["delta_cell_recall@5_vs_flat"]) >= 0.01 and
            row["delta_worst_client_vs_flat"] >= -0.005)
    candidates = [row for row in rows if row["gate_family"].startswith("baseline_") is False]
    best = max(candidates, key=lambda row: (row["weighted_ndcg@10"],
                                            row["worst_client_weighted_ndcg@10"],
                                            -row["unitable_queries"]))
    best_config = next(config for config in configs if config["config_id"] == best["config_id"])
    best_payload = {
        "schema_version": "1.0", "task": 4, "selection_split": "dev",
        "test_accessed": False, "seed": args.seed,
        "strategy": "confidence_gating", "config": best_config,
        "composite_r_formula": COMPOSITE_WEIGHTS, "metrics": best,
        "flat_dev_baseline": baseline,
        "selection_rule": "maximize dev weighted NDCG@10; tie-break worst-client then fewer UniTable queries",
        "eligible_for_task6_gate": bool(best["passes_preregistered_gate"]),
        "note": "Task 4 alone does not authorize formal test; Task 5 and Task 6 remain required.",
    }
    output_dir = args.output_dir
    grid_path = output_dir / "gating_dev_grid.csv"
    best_path = output_dir / "gating_dev_best_config.json"
    summary_path = output_dir / "gating_dev_summary.md"
    manifest_path = output_dir / "gating_sha256_manifest.json"
    atomic_csv(grid_path, rows, args.overwrite)
    atomic_text(best_path, json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    status = "达到任务4数值门槛；仍须完成任务5和任务6" if best["passes_preregistered_gate"] else (
        "未达到预注册门槛；不得据任务4结果运行test")
    summary = f"""# Task 4 冻结 BGE 置信度门控（dev-only）

## 协议

- 仅使用 dev 的冻结 Flat/UniTable 精确检索结果；未训练、未重新编码、未读取 test。
- 门控对每个 query 统一选择整套 Flat 或 UniTable 三层结果。
- 综合可靠性 R = 0.40×S-TEDS + 0.25×cell-count consistency + 0.15×valid_html + 0.20×confidence。
- 层级权重：Table 0.2、Row 0.3、Cell 0.5；公平性使用冻结 5 客户端划分。

## 结果

- 网格配置数：{len(rows)}；dev query 数：{len(flat)}。
- Flat baseline weighted NDCG@10：{baseline['weighted_ndcg@10']:.9f}。
- 最佳门控：{best['config_id']} / {best['gate_family']}，UniTable query={best['unitable_queries']} ({best['unitable_fraction']:.2%})。
- 最佳 weighted NDCG@10：{best['weighted_ndcg@10']:.9f}，相对 Flat {best['delta_weighted_ndcg_vs_flat']:+.9f}。
- Row Recall@5 变化：{best['delta_row_recall@5_vs_flat']:+.9f}；Cell Recall@5 变化：{best['delta_cell_recall@5_vs_flat']:+.9f}。
- worst-client weighted NDCG@10 变化：{best['delta_worst_client_vs_flat']:+.9f}。
- 判定：{status}。

## 边界

本结果只用于 dev 方案比较。无论任务4单项结果如何，均不得直接进入正式 test；须继续完成任务5，并由任务6统一冻结或输出 fail decision。
"""
    atomic_text(summary_path, summary, args.overwrite)
    manifest = {
        "schema_version": "1.0", "selection_split": "dev", "test_accessed": False,
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in (args.flat_dev, args.unitable_dev, args.pairwise_dev,
                                args.structure_metrics, args.evidence_map, args.partition)],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in (grid_path, best_path, summary_path)],
    }
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    LOG.info("Task 4 complete configs=%d best=%s weighted_ndcg=%.9f", len(rows),
             best["config_id"], best["weighted_ndcg@10"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Task 4 confidence gating failed")
        sys.exit(2)
