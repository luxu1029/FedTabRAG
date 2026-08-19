#!/usr/bin/env python3
"""Finalize the frozen Day-8 formal-test audit and comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from pathlib import Path


RUNS = {
    "fedavg": ("day75_queue_lr1e5_pilot_s42",
               "formal_test_best_round4_once.json"),
    "r_only": ("day8_r_only_s42", "formal_test_best_round4_once.json"),
    "u_only": ("day8_u_only_s42", "formal_test_best_round4_once.json"),
    "r_plus_u": ("day8_r_plus_u_s42", "formal_test_best_round4_once.json"),
}
LEVELS = ("table", "row", "cell")
LEVEL_WEIGHTS = {"table": 0.2, "row": 0.3, "cell": 0.5}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/federated"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dev = json.loads((args.root / "day8_dev_summary.json").read_text())
    result = {
        "seed": args.seed, "selection_split": "dev",
        "test_policy": "one evaluation per frozen best; no test-based tuning",
        "baselines": {
            "day75_dev_best": 0.23749736594849535,
            "unitable_test_weighted_ndcg": 0.229913557967246,
            "flat_test_weighted_ndcg": 0.256153879,
        },
        "runs": {},
    }
    for mode, (name, test_name) in RUNS.items():
        run = args.root / name
        test_path = run / test_name
        payload = json.loads(test_path.read_text())
        if payload["checkpoint_round"] != dev["runs"][mode]["best_round"]:
            raise ValueError(f"{mode}: formal test did not use frozen dev best")
        global_weighted = sum(
            LEVEL_WEIGHTS[level] * payload["global"][level]["ndcg@10"]
            for level in LEVELS)
        client_ids = sorted(payload["clients"]["table"])
        per_client = {
            cid: sum(LEVEL_WEIGHTS[level] *
                     payload["clients"][level][cid]["ndcg@10"]
                     for level in LEVELS)
            for cid in client_ids
        }
        client_values = list(per_client.values())
        frozen = run / "frozen_day8_best_round4"
        manifest_path = frozen / "freeze_manifest.json"
        manifest_path.chmod(0o644)
        manifest = json.loads(manifest_path.read_text())
        manifest["formal_test_evaluations"] = 1
        manifest["formal_test_result"] = {
            "path": f"../{test_name}", "sha256": sha256(test_path),
            "bytes": test_path.stat().st_size,
            "checkpoint_round": payload["checkpoint_round"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.chmod(0o444)
        test_path.chmod(0o444)
        result["runs"][mode] = {
            **dev["runs"][mode],
            "test": {
                "table_ndcg@10": payload["global"]["table"]["ndcg@10"],
                "row_ndcg@10": payload["global"]["row"]["ndcg@10"],
                "cell_ndcg@10": payload["global"]["cell"]["ndcg@10"],
                "weighted_ndcg@10": global_weighted,
                "macro_weighted_ndcg@10": statistics.fmean(client_values),
                "worst_client_weighted_ndcg@10": min(client_values),
                "std_client_weighted_ndcg@10": statistics.pstdev(client_values),
                "per_client_weighted_ndcg@10": per_client,
                "sha256": sha256(test_path),
            },
        }
    fedavg = result["runs"]["fedavg"]["test"]["weighted_ndcg@10"]
    for mode, row in result["runs"].items():
        row["test"]["delta_vs_fedavg"] = row["test"]["weighted_ndcg@10"] - fedavg
    result["decision"] = {
        "dev_primary_improved": (
            result["runs"]["u_only"]["dev_weighted_ndcg"] >
            result["baselines"]["day75_dev_best"]),
        "dev_improvement_stable_two_rounds": False,
        "row_dev_clearly_improved": False,
        "worst_client_dev_improved": False,
        "passed_day8_gate": False,
        "action": (
            "pause expensive Day-9 training; diagnose row candidates, "
            "multi-positive weighting, and client sampling bias"),
    }
    output = args.root / "day8_final_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    logging.info("wrote %s", output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-8 summary failed")
        raise
