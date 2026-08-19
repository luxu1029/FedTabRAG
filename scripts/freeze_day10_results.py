#!/usr/bin/env python3
"""Freeze Day-10 configs/results with SHA256 and create a weight-free archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".gguf", ".ckpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/day10"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed(path: Path, output_root: Path) -> bool:
    if not path.is_file() or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name in {"FedTabRAG_day10_frozen_results.tar.gz", "package_sha256.txt"}:
        return False
    if path.name == "corpus_test.jsonl" and "perturbations" in path.parts:
        return False
    return not any(part in {"checkpoints", "models", "weights"} for part in path.parts)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = (root / args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    candidates: set[Path] = set()
    candidates.update(path for path in out.rglob("*") if allowed(path, out))
    for pattern in ("configs/federated/day10_*.yaml", "scripts/*day10*.py"):
        candidates.update(path for path in root.glob(pattern) if allowed(path, out))
    candidates.update(path for path in (
        root / "scripts/build_day10_perturbations.py",
        root / "src/federated/evaluate_fedrag.py",
        root / "outputs/day9/ablation_summary.json",
        root / "outputs/unitable/structure_metrics.json",
    ) if allowed(path, out))
    for run_name in ("fedrag_flat_s42", "fedrag_unitable_s42", "day8_r_plus_u_s42",
                     "day10_flat_s7", "day10_unitable_s7", "day10_r_plus_u_s7",
                     "day10_flat_s2026", "day10_unitable_s2026", "day10_r_plus_u_s2026"):
        run = root / "outputs/federated" / run_name
        for name in ("convergence.csv", "training_audit.json", "resolved_config.json",
                     "test_metrics.json", "formal_test_best_round4_once.json"):
            path = run / name
            if allowed(path, out):
                candidates.add(path)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "git_commit": "NA",
        "git_commit_reason": "workspace is not a Git repository",
        "weight_files_included": 0,
        "excluded": ["model/checkpoint weights", "perturbation corpus copies"],
    }
    metadata_path = out / "freeze_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    candidates.add(metadata_path)

    # The manifest intentionally excludes itself and the archive to avoid recursive hashes.
    entries = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256(path)} for path in sorted(candidates)]
    manifest_path = out / "sha256_manifest.json"
    manifest_path.write_text(json.dumps({"algorithm": "SHA256", "files": entries},
                                        indent=2) + "\n")
    candidates.add(manifest_path)

    archive = out / "FedTabRAG_day10_frozen_results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(candidates):
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    archive_hash = sha256(archive)
    (out / "package_sha256.txt").write_text(
        f"{archive_hash}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "archive_sha256": archive_hash,
                      "file_count": len(candidates), "weight_files_included": 0},
                     indent=2))


if __name__ == "__main__":
    main()
