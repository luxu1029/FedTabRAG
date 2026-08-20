#!/usr/bin/env python3
"""Select and freeze a structure strategy using dev metrics only.

Task 6 is a decision layer: it never retrieves, encodes, trains, or reads test
artifacts. All candidates are judged against one canonical Flat dev baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any


LOG = logging.getLogger("select_and_freeze_structure_strategy")
THRESHOLDS = {
    "weighted_ndcg_min_improvement": 0.003,
    "row_or_cell_recall5_min_improvement": 0.01,
    "worst_client_max_degradation": 0.005,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gating-grid", type=Path, default=Path(
        "outputs/diagnosis/gating/gating_dev_grid.csv"))
    parser.add_argument("--gating-best", type=Path, default=Path(
        "outputs/diagnosis/gating/gating_dev_best_config.json"))
    parser.add_argument("--gating-manifest", type=Path, default=Path(
        "outputs/diagnosis/gating/gating_sha256_manifest.json"))
    parser.add_argument("--fusion-grid", type=Path, default=Path(
        "outputs/diagnosis/fusion/fusion_dev_grid.csv"))
    parser.add_argument("--fusion-best", type=Path, default=Path(
        "outputs/diagnosis/fusion/fusion_dev_best_config.json"))
    parser.add_argument("--fusion-manifest", type=Path, default=Path(
        "outputs/diagnosis/fusion/fusion_sha256_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/diagnosis/frozen_selection"))
    parser.add_argument("--log-file", type=Path, default=Path(
        "outputs/logs/diagnosis_frozen_selection.log"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty grid: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("selection_split") != "dev" or manifest.get("test_accessed") is not False:
        raise ValueError(f"Manifest is not dev-only: {path}")
    for item in manifest.get("outputs", []):
        output = Path(item["path"])
        if sha256(output) != item["sha256"]:
            raise ValueError(f"Upstream SHA256 mismatch: {output}")
    return manifest


def as_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "weighted_ndcg@10": float(row["weighted_ndcg@10"]),
        "row_recall@5": float(row["row_recall@5"]),
        "cell_recall@5": float(row["cell_recall@5"]),
        "worst_client_weighted_ndcg@10": float(row["worst_client_weighted_ndcg@10"]),
    }


def judge(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    deltas = {
        "weighted_ndcg@10": metrics["weighted_ndcg@10"] - baseline["weighted_ndcg@10"],
        "row_recall@5": metrics["row_recall@5"] - baseline["row_recall@5"],
        "cell_recall@5": metrics["cell_recall@5"] - baseline["cell_recall@5"],
        "worst_client_weighted_ndcg@10": (
            metrics["worst_client_weighted_ndcg@10"] -
            baseline["worst_client_weighted_ndcg@10"]),
    }
    checks = {
        "weighted_ndcg": deltas["weighted_ndcg@10"] >=
                         THRESHOLDS["weighted_ndcg_min_improvement"],
        "row_or_cell_recall5": max(deltas["row_recall@5"], deltas["cell_recall@5"]) >=
                               THRESHOLDS["row_or_cell_recall5_min_improvement"],
        "worst_client": deltas["worst_client_weighted_ndcg@10"] >=
                        -THRESHOLDS["worst_client_max_degradation"],
    }
    return {"deltas_vs_canonical_flat": deltas, "checks": checks,
            "passes": all(checks.values())}


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    forbidden = [path for path in (args.gating_grid, args.gating_best, args.gating_manifest,
                                   args.fusion_grid, args.fusion_best, args.fusion_manifest)
                 if "test" in path.name.lower()]
    if forbidden:
        raise ValueError(f"Test-like input paths are forbidden: {forbidden}")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter); stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    LOG.info("Starting Task 6 seed=%d; dev-only selection", args.seed)

    validate_manifest(args.gating_manifest); validate_manifest(args.fusion_manifest)
    gating_best, fusion_best = load_json(args.gating_best), load_json(args.fusion_best)
    for label, payload in (("gating", gating_best), ("fusion", fusion_best)):
        if payload.get("selection_split") != "dev" or payload.get("test_accessed") is not False:
            raise ValueError(f"{label} best config is not dev-only")
    gating_rows, fusion_rows = read_csv(args.gating_grid), read_csv(args.fusion_grid)
    if len({row["config_id"] for row in gating_rows}) != len(gating_rows):
        raise ValueError("Duplicate gating config_id")
    if len({row["config_id"] for row in fusion_rows}) != len(fusion_rows):
        raise ValueError("Duplicate fusion config_id")
    flat_row = next((row for row in gating_rows if row["gate_family"] == "baseline_flat"), None)
    if flat_row is None:
        raise ValueError("Canonical Flat baseline missing from gating grid")
    baseline = as_metrics(flat_row)

    candidates: list[dict[str, Any]] = []
    none = {"strategy": "none", "config_id": flat_row["config_id"],
            "metrics": baseline, "protocol_comparable_to_canonical_flat": True}
    none.update(judge(baseline, baseline)); candidates.append(none)
    for row in gating_rows:
        if row["gate_family"].startswith("baseline_"):
            continue
        candidate = {
            "strategy": "gating", "config_id": row["config_id"],
            "parameters": {
                key: row[key] for key in (
                    "gate_family", "s_teds_threshold",
                    "cell_count_consistency_threshold", "require_valid_html",
                    "confidence_threshold", "composite_r_threshold")
            },
            "metrics": as_metrics(row), "protocol_comparable_to_canonical_flat": True,
        }
        candidate.update(judge(candidate["metrics"], baseline)); candidates.append(candidate)
    for row in fusion_rows:
        candidate = {
            "strategy": "fusion", "config_id": row["config_id"],
            "parameters": {"normalization": row["normalization"],
                           "alpha": float(row["alpha"]),
                           "candidate_alignment": row["candidate_alignment"]},
            "metrics": as_metrics(row),
            "protocol_comparable_to_canonical_flat": False,
            "comparability_note": (
                "Frozen Top-10 union uses two non-overlapping row/cell representation-ID pools; "
                "absolute NDCG is not protocol-identical to canonical full-corpus Flat."),
        }
        candidate.update(judge(candidate["metrics"], baseline))
        # Even if raw thresholds passed, a non-comparable metric cannot freeze a formal-test config.
        candidate["raw_thresholds_pass"] = candidate["passes"]
        candidate["passes"] = bool(candidate["passes"] and
                                     candidate["protocol_comparable_to_canonical_flat"])
        candidates.append(candidate)
    candidates.append({
        "strategy": "gating_plus_fusion", "config_id": None, "metrics": None,
        "protocol_comparable_to_canonical_flat": False, "passes": False,
        "unavailable_reason": (
            "No independently evaluated dev grid exists for the composed strategy. Task 6 does "
            "not invent metrics or search a new composition."),
    })
    passing = [candidate for candidate in candidates if candidate.get("passes")]
    selected = (max(passing, key=lambda item: (
        item["metrics"]["weighted_ndcg@10"],
        item["metrics"]["worst_client_weighted_ndcg@10"])) if passing else None)

    output_dir = args.output_dir
    frozen_path = output_dir / "frozen_config.json"
    fail_path = output_dir / "fail_decision.json"
    report_path = output_dir / "dev_selection_report.md"
    manifest_path = output_dir / "freeze_manifest.json"
    if selected is None and frozen_path.exists():
        raise FileExistsError(f"Stale pass config exists and will not be deleted automatically: {frozen_path}")
    if selected is not None and fail_path.exists():
        raise FileExistsError(f"Stale fail decision exists and will not be deleted automatically: {fail_path}")

    best_gating = max((item for item in candidates if item["strategy"] == "gating"),
                      key=lambda item: item["metrics"]["weighted_ndcg@10"])
    best_fusion = max((item for item in candidates if item["strategy"] == "fusion"),
                      key=lambda item: item["metrics"]["weighted_ndcg@10"])
    if selected is None:
        decision = {
            "schema_version": "1.0", "task": 6, "decision": "FAIL_NO_DEV_STRATEGY",
            "selection_split": "dev", "test_accessed": False,
            "formal_test_authorized": False, "frozen_config_created": False,
            "canonical_flat_dev_baseline": baseline, "thresholds": THRESHOLDS,
            "candidate_counts": {"none": 1,
                                 "gating": sum(x["strategy"] == "gating" for x in candidates),
                                 "fusion": sum(x["strategy"] == "fusion" for x in candidates),
                                 "gating_plus_fusion": 1},
            "passing_candidate_count": 0,
            "best_gating": best_gating, "best_fusion_by_weighted_ndcg": best_fusion,
            "gating_plus_fusion": candidates[-1],
            "reasons": [
                "No gating candidate improves canonical Flat weighted NDCG@10 by at least 0.003 while satisfying recall and worst-client constraints.",
                "Fusion Top-10-union metrics are not protocol-comparable to canonical full-corpus Flat; when numerically compared to the canonical baseline, no fusion candidate passes all thresholds.",
                "No dev result exists for gating_plus_fusion, so Task 6 cannot freeze that strategy.",
            ],
            "required_action": "Do not run Task 7 formal test. Freeze the negative dev decision.",
        }
        atomic_text(fail_path, json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                    args.overwrite)
        decision_output = fail_path
    else:
        frozen = {
            "schema_version": "1.0", "task": 6, "selection_split": "dev",
            "test_accessed": False, "formal_test_authorized": True,
            "canonical_flat_dev_baseline": baseline, "thresholds": THRESHOLDS,
            "selected": selected,
        }
        atomic_text(frozen_path, json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
                    args.overwrite)
        decision_output = frozen_path

    bg, bf = best_gating["metrics"], best_fusion["metrics"]
    conclusion = ("PASS：已冻结可进入一次正式test的配置。" if selected else
                  "FAIL_NO_DEV_STRATEGY：不得进入任务7正式test。")
    action_text = ("已生成frozen_config.json；任务7只允许按该配置进行一次正式test。"
                   if selected else
                   "未生成可供任务7读取的frozen_config.json。已生成fail_decision.json并冻结失败原因。后续不得运行正式test；按工作安排应停止本轮结构策略test放行，并进入负结果整理/问答端方向。")
    report = f"""# Task 6 dev-only 结构策略选择与冻结报告

## 协议

- 选择只读取任务4/5的dev网格、最佳配置和SHA256清单；未读取test、未训练、未重新编码。
- 所有候选统一对比canonical Flat dev：weighted NDCG@10={baseline['weighted_ndcg@10']:.9f}，Row Recall@5={baseline['row_recall@5']:.9f}，Cell Recall@5={baseline['cell_recall@5']:.9f}，worst-client={baseline['worst_client_weighted_ndcg@10']:.9f}。
- 门槛：Δweighted NDCG≥0.003；ΔRow或ΔCell Recall@5≥0.01；Δworst-client≥-0.005。

## 候选审查

- none：不提升，未过门槛。
- gating最佳 {best_gating['config_id']}：weighted NDCG={bg['weighted_ndcg@10']:.9f}，Δ={best_gating['deltas_vs_canonical_flat']['weighted_ndcg@10']:+.9f}，未过门槛。
- fusion最高 {best_fusion['config_id']}：池内weighted NDCG={bf['weighted_ndcg@10']:.9f}；相对canonical Flat Δ={best_fusion['deltas_vs_canonical_flat']['weighted_ndcg@10']:+.9f}，且Top-10 union口径与全语料Flat不完全可比，未过门槛。
- gating_plus_fusion：没有独立dev网格；不伪造组合指标，记为不可评估/未通过。

## 冻结决定

{conclusion}

{action_text}
"""
    atomic_text(report_path, report, args.overwrite)
    script_path = Path(__file__).resolve()
    inputs = (args.gating_grid, args.gating_best, args.gating_manifest,
              args.fusion_grid, args.fusion_best, args.fusion_manifest, script_path)
    outputs = (decision_output, report_path)
    manifest = {
        "schema_version": "1.0", "task": 6, "selection_split": "dev",
        "test_accessed": False, "decision": "pass" if selected else "fail",
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in inputs],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in outputs],
        "manifest_self_hash_excluded": True,
    }
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    LOG.info("Task 6 complete decision=%s passing=%d", "pass" if selected else "fail",
             len(passing))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Task 6 selection/freezing failed")
        sys.exit(2)
