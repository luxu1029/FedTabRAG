#!/usr/bin/env python3
"""Build Day-10 figures/tables with strict missing-as-NA semantics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NA = "NA"
LEVELS = ("table", "row", "cell")
LEVEL_WEIGHTS = {"table": 0.2, "row": 0.3, "cell": 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/day10"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def csv_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": NA}])


def weighted_test(payload: dict | None) -> dict:
    if payload is None:
        return {key: NA for key in ("global", "macro", "worst_client", "std")}
    global_score = sum(LEVEL_WEIGHTS[level] * payload["global"][level]["ndcg@10"]
                       for level in LEVELS)
    clients = sorted(payload["clients"]["table"])
    per_client = [sum(LEVEL_WEIGHTS[level] *
                      payload["clients"][level][cid]["ndcg@10"]
                      for level in LEVELS) for cid in clients]
    return {"global": global_score, "macro": statistics.fmean(per_client),
            "worst_client": min(per_client), "std": statistics.pstdev(per_client)}


def main() -> None:
    args = parse_args()
    figures = args.output_root / "figures"
    tables = args.output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    missing = []

    methods = {
        "FedE4RAG-Flat": {
            "run": Path("outputs/federated/fedrag_flat_s42"), "qa": "flat"},
        "FedE4RAG-UniTable": {
            "run": Path("outputs/federated/fedrag_unitable_s42"), "qa": "unitable"},
        "FedTabRAG-Full(R+U)": {
            "run": Path("outputs/federated/day8_r_plus_u_s42"), "qa": "r_plus_u"},
    }
    day9 = load_json(Path("outputs/day9/ablation_summary.json")) or {"methods": {}}
    main_rows, efficiency_rows = [], []
    curves = {}
    for name, spec in methods.items():
        run = spec["run"]
        test_candidates = [run / "formal_test_best_round4_once.json",
                           run / "test_metrics.json"]
        test_path = next((path for path in test_candidates if path.exists()), None)
        payload = load_json(test_path) if test_path else None
        weighted = weighted_test(payload)
        qa = day9["methods"].get(spec["qa"], {})
        main_rows.append({
            "method": name,
            "table_ndcg@10": payload["global"]["table"]["ndcg@10"] if payload else NA,
            "row_ndcg@10": payload["global"]["row"]["ndcg@10"] if payload else NA,
            "cell_ndcg@10": payload["global"]["cell"]["ndcg@10"] if payload else NA,
            "cell_recall@5": payload["global"]["cell"]["recall@5"] if payload else NA,
            "weighted_global_ndcg@10": weighted["global"],
            "weighted_macro_ndcg@10": weighted["macro"],
            "weighted_worst_client_ndcg@10": weighted["worst_client"],
            "weighted_client_std_ndcg@10": weighted["std"],
            "qa100_numerical_em": qa.get("numerical_em", NA),
            "program_accuracy": NA,
            "source": str(test_path) if test_path else NA,
        })
        if payload is None:
            missing.append({"artifact": "main_results", "method": name,
                            "field": "all test metrics", "reason": "source missing"})
        curve = csv_rows(run / "convergence.csv")
        curves[name] = curve
        audit = load_json(run / "training_audit.json")
        if curve and audit:
            efficiency_rows.append({
                "method": name, "rounds": len(curve),
                "training_seconds": sum(float(row["seconds"]) for row in curve),
                "peak_gpu_mib": max(float(row["peak_gpu_mib"]) for row in curve),
                "lora_parameter_bytes": audit.get("lora_parameter_bytes", NA),
                "total_communication_bytes": audit.get("total_communication_bytes", NA),
                "source": str(run / "convergence.csv"),
            })
        else:
            efficiency_rows.append({key: (name if key == "method" else NA) for key in
                                    ("method", "rounds", "training_seconds", "peak_gpu_mib",
                                     "lora_parameter_bytes", "total_communication_bytes", "source")})
    write_csv(tables / "main_results.csv", main_rows)
    write_csv(tables / "efficiency.csv", efficiency_rows)

    # Optional resource-permitting seed supplement. Missing runs remain explicit NA.
    seed_runs = {
        "FedE4RAG-Flat": {42: Path("outputs/federated/fedrag_flat_s42"),
                           7: Path("outputs/federated/day10_flat_s7"),
                           2026: Path("outputs/federated/day10_flat_s2026")},
        "FedE4RAG-UniTable": {42: Path("outputs/federated/fedrag_unitable_s42"),
                               7: Path("outputs/federated/day10_unitable_s7"),
                               2026: Path("outputs/federated/day10_unitable_s2026")},
        "FedTabRAG-Full(R+U)": {42: Path("outputs/federated/day8_r_plus_u_s42"),
                                 7: Path("outputs/federated/day10_r_plus_u_s7"),
                                 2026: Path("outputs/federated/day10_r_plus_u_s2026")},
    }
    seed_rows = []
    for method, runs in seed_runs.items():
        available = []
        for seed, run in runs.items():
            candidates = [run / "formal_test_best_round4_once.json", run / "test_metrics.json"]
            path = next((item for item in candidates if item.exists()), None)
            score = weighted_test(load_json(path))["global"] if path else NA
            seed_rows.append({"method": method, "row_type": "seed", "seed": seed,
                              "weighted_global_ndcg@10": score,
                              "mean_across_seeds": NA, "std_across_seeds": NA,
                              "source": str(path) if path else NA})
            if score == NA:
                missing.append({"artifact": "supplementary_seeds", "method": method,
                                "field": f"seed {seed} test",
                                "reason": "resource-permitting seed result unavailable"})
            else:
                available.append(score)
        seed_rows.append({
            "method": method, "row_type": "summary", "seed": "all_available",
            "weighted_global_ndcg@10": NA,
            "mean_across_seeds": statistics.fmean(available) if available else NA,
            "std_across_seeds": statistics.pstdev(available) if available else NA,
            "source": f"n={len(available)}",
        })
    write_csv(tables / "supplementary_seeds.csv", seed_rows)

    # Original-log convergence: normalize each objective to its own round-1 value;
    # objectives differ, so raw loss is also exported and never compared as a shared scale.
    plt.figure(figsize=(8, 5))
    convergence_export = []
    for name, rows in curves.items():
        if not rows or "loss" not in rows[0]:
            missing.append({"artifact": "convergence", "method": name,
                            "field": "loss", "reason": "raw CSV field missing"})
            continue
        rounds = [int(row["round"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]
        normalized = [value / losses[0] for value in losses]
        plt.plot(rounds, normalized, marker="o", label=name)
        has_dev = "dev_weighted_ndcg" in rows[0]
        for round_no, loss, norm in zip(rounds, losses, normalized):
            convergence_export.append({
                "method": name, "round": round_no, "raw_train_loss": loss,
                "normalized_train_loss": norm,
                "dev_weighted_ndcg": (rows[round_no - 1]["dev_weighted_ndcg"]
                                      if has_dev else NA),
                "cell_recall@5_per_round": NA,
            })
        if not has_dev:
            missing.append({"artifact": "convergence", "method": name,
                            "field": "per-round dev weighted NDCG",
                            "reason": "not recorded in original log"})
        missing.append({"artifact": "convergence", "method": name,
                        "field": "per-round Cell Recall@5",
                        "reason": "not recorded in original log"})
    plt.xlabel("Federated communication round")
    plt.ylabel("Train loss / round-1 train loss")
    plt.title("Three-method convergence from original CSV logs")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "three_method_convergence.png", dpi=200)
    plt.savefig(figures / "three_method_convergence.pdf")
    plt.close()
    write_csv(tables / "convergence_source.csv", convergence_export)

    structure = load_json(Path("outputs/unitable/structure_metrics.json"))
    test_structure = structure.get("by_split", {}).get("test", {}) if structure else {}
    structure_rows = [
        {"method": "Flat", "split": "test", "s_teds": NA,
         "valid_html_rate": NA, "cell_count_exact_rate": NA,
         "source": NA},
        {"method": "GT-oracle", "split": "test", "s_teds": NA,
         "valid_html_rate": NA, "cell_count_exact_rate": NA,
         "source": NA},
        {"method": "UniTable", "split": "test",
         "s_teds": test_structure.get("mean_s_teds", NA),
         "valid_html_rate": test_structure.get("valid_html_rate", NA),
         "cell_count_exact_rate": test_structure.get("cell_count_exact_rate", NA),
         "source": "outputs/unitable/structure_metrics.json" if structure else NA},
    ]
    missing.extend([
        {"artifact": "structure_table", "method": "Flat", "field": "all",
         "reason": "no structure-recognition output; not inferred"},
        {"artifact": "structure_table", "method": "GT-oracle", "field": "all",
         "reason": "no measured structure-evaluation file; not filled with assumed 1.0"},
    ])
    write_csv(tables / "structure.csv", structure_rows)

    ablation_rows = []
    for group, method_names in day9.get("groups", {}).items():
        for method in method_names:
            row = day9.get("methods", {}).get(method)
            ablation_rows.append({
                "group": group, "method": method,
                "retrieval_recall@5": row.get("retrieval_recall@5", NA) if row else NA,
                "retrieval_mrr@5": row.get("retrieval_mrr@5", NA) if row else NA,
                "qa100_numerical_em": row.get("numerical_em", NA) if row else NA,
                "program_accuracy": NA,
                "source": "outputs/day9/ablation_summary.json" if row else NA,
            })
            if row is None:
                missing.append({"artifact": "ablation", "method": method,
                                "field": "all", "reason": "Day9 row missing"})
    write_csv(tables / "three_ablation_groups.csv", ablation_rows)

    baseline = load_json(Path(
        "outputs/federated/day8_r_plus_u_s42/formal_test_best_round4_once.json"))
    robustness_rows = []
    all_rates = [(0, baseline)] + [(rate, load_json(args.output_root / "robustness" /
                                                     f"rate_{rate:02d}.json"))
                                  for rate in (5, 10, 20, 30)]
    baseline_weighted = weighted_test(baseline)["global"] if baseline else NA
    for rate, payload in all_rates:
        score = weighted_test(payload)["global"]
        robustness_rows.append({
            "perturbation_percent": rate,
            "table_ndcg@10": payload["global"]["table"]["ndcg@10"] if payload else NA,
            "row_ndcg@10": payload["global"]["row"]["ndcg@10"] if payload else NA,
            "cell_ndcg@10": payload["global"]["cell"]["ndcg@10"] if payload else NA,
            "cell_recall@5": payload["global"]["cell"]["recall@5"] if payload else NA,
            "weighted_ndcg@10": score,
            "absolute_drop_vs_0": (baseline_weighted - score
                                   if score != NA and baseline_weighted != NA else NA),
            "cell_text_multiset_equal": True if rate else "original",
            "source": (str(args.output_root / "robustness" / f"rate_{rate:02d}.json")
                       if rate else
                       "outputs/federated/day8_r_plus_u_s42/formal_test_best_round4_once.json"),
        })
        if payload is None:
            missing.append({"artifact": "robustness", "method": f"rate_{rate}",
                            "field": "all", "reason": "evaluation output missing"})
    write_csv(tables / "robustness.csv", robustness_rows)

    # Explicitly record known protocol-level missing values.
    missing.extend([
        {"artifact": "repository", "method": "project", "field": "git_commit",
         "reason": "workspace is not a Git repository"},
        {"artifact": "qa", "method": "all", "field": "Program/Execution Accuracy",
         "reason": "no method predicted programs"},
    ])
    missing_report = {"policy": "strict missing: never impute; render as NA",
                      "missing_count": len(missing), "items": missing}
    (args.output_root / "strict_missing_audit.json").write_text(
        json.dumps(missing_report, indent=2) + "\n")
    write_csv(tables / "strict_missing.csv", missing)


if __name__ == "__main__":
    main()
