#!/usr/bin/env python3
"""Run the three preregistered Day-10 methods for one additional seed."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    logging.info("running %s", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed = int(yaml.safe_load(args.config.read_text())["seed"])
    py = sys.executable
    run([py, "src/federated/train_fedrag.py", "--config", str(args.config),
         "--variant", "flat", "--run-name", f"day10_flat_s{seed}"])
    run([py, "src/federated/train_fedrag.py", "--config", str(args.config),
         "--variant", "predicted_structure", "--run-name", f"day10_unitable_s{seed}"])
    run([py, "src/federated/train_fedtabrag.py", "--config", str(args.config),
         "--features", "hierarchy,hard_negative,hierarchical_kd,local_queue",
         "--run-name", f"day10_r_plus_u_s{seed}", "--stage", "full",
         "--rounds", "10", "--queue-size", "128", "--learning-rate", "1e-5",
         "--select-by", "dev_weighted_ndcg", "--aggregation", "r_plus_u",
         "--aggregation-min-weight", "0.05", "--aggregation-max-weight", "0.5"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-10 seed run failed")
        raise
