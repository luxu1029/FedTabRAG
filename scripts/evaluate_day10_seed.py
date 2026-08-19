#!/usr/bin/env python3
"""Evaluate the three frozen best checkpoints for one Day-10 seed exactly once."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--partition", type=Path,
                        default=Path("data/processed/federated/company_5clients.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for method, variant in (("flat", "flat"), ("unitable", "predicted_structure"),
                            ("r_plus_u", "predicted_structure")):
        run = Path(f"outputs/federated/day10_{method}_s{args.seed}")
        output = run / "test_metrics.json"
        if output.exists():
            raise FileExistsError(f"refusing to repeat/overwrite formal test: {output}")
        command = [sys.executable, "src/federated/evaluate_fedrag.py",
                   "--run-dir", str(run), "--checkpoint", "best",
                   "--variant", variant, "--partition", str(args.partition),
                   "--output", str(output), "--device", args.device,
                   "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
        logging.info("running %s", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-10 seed evaluation failed")
        raise
