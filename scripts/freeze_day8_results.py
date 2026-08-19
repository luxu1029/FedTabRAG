#!/usr/bin/env python3
"""Freeze Day-8 dev-selected checkpoints before any formal test evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path

import torch


RUNS = {
    "fedavg": "day75_queue_lr1e5_pilot_s42",
    "r_only": "day8_r_only_s42",
    "u_only": "day8_u_only_s42",
    "r_plus_u": "day8_r_plus_u_s42",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/federated"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = {
        "seed": args.seed, "selection_split": "dev",
        "selection_metric": "0.2*table_ndcg@10+0.3*row_ndcg@10+0.5*cell_ndcg@10",
        "formal_test_evaluations_at_freeze": 0, "runs": {},
    }
    for mode, name in RUNS.items():
        run = args.output_root / name
        rows = list(csv.DictReader((run / "convergence.csv").open()))
        best_row = max(rows, key=lambda row: float(row["dev_weighted_ndcg"]))
        checkpoint = torch.load(
            run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        best_round = int(best_row["round"])
        if int(checkpoint["round"]) != best_round:
            raise ValueError(f"{name}: curve/checkpoint best mismatch")
        frozen = run / f"frozen_day8_best_round{best_round}"
        frozen.mkdir(exist_ok=True)
        sources = {
            "checkpoints/best.pt": run / "checkpoints/best.pt",
            "resolved_config.json": run / "resolved_config.json",
            "convergence.csv": run / "convergence.csv",
            "training_audit.json": run / "training_audit.json",
            f"dev_metrics_round_{best_round}.json":
                run / f"dev_metrics_round_{best_round}.json",
        }
        for target_name, source in sources.items():
            target = frozen / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
        manifest = {
            "mode": mode, "run": name, "seed": args.seed,
            "selection_split": "dev", "best_round": best_round,
            "best_dev_weighted_ndcg": float(best_row["dev_weighted_ndcg"]),
            "test_evaluations_before_freeze": (
                1 if mode == "fedavg" else
                len(list(run.glob("*test*.json")))),
            "files": {
                target_name: {
                    "sha256": sha256(frozen / target_name),
                    "bytes": (frozen / target_name).stat().st_size,
                }
                for target_name in sources
            },
        }
        manifest_path = frozen / "freeze_manifest.json"
        if manifest_path.exists():
            manifest_path.chmod(0o644)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for path in frozen.rglob("*"):
            if path.is_dir():
                continue
            path.chmod(0o444)
        summary["runs"][mode] = {
            "run": name, "best_round": best_round,
            "dev_weighted_ndcg": float(best_row["dev_weighted_ndcg"]),
            "dev_table_ndcg": float(best_row["dev_table_ndcg"]),
            "dev_row_ndcg": float(best_row["dev_row_ndcg"]),
            "dev_cell_ndcg": float(best_row["dev_cell_ndcg"]),
            "dev_worst_ndcg": float(best_row["dev_worst_ndcg"]),
            "checkpoint_sha256": manifest["files"]["checkpoints/best.pt"]["sha256"],
            "frozen_dir": str(frozen),
        }
        logging.info("froze mode=%s round=%d", mode, best_round)
    output = args.output_root / "day8_dev_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("wrote %s", output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-8 freeze failed")
        raise
