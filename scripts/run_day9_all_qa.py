#!/usr/bin/env python3
"""Run all frozen Day-9 QA methods sequentially with per-method resume."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


METHODS = (
    "flat", "gt_zero_shot", "unitable", "hierarchy", "hierarchy_hard",
    "hierarchy_hard_kd", "fedavg_queue", "r_only", "u_only", "r_plus_u",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    script = Path(__file__).with_name("run_day9_qa.py")
    for method in METHODS:
        logging.info("starting method=%s", method)
        subprocess.run([
            sys.executable, str(script), "--method", method,
            "--ollama-url", args.ollama_url, "--seed", str(args.seed),
        ], check=True)
    logging.info("all methods completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-9 all-method QA failed")
        raise
