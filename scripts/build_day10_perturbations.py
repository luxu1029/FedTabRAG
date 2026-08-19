#!/usr/bin/env python3
"""Create nested structural-perturbation copies with invariant cell-text multisets."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/predicted_structure/corpus_test.jsonl"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/day10/perturbations"))
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def perturb_sample(rows: list[dict], operation: str) -> bool:
    cells = [row for row in rows if row.get("level") == "cell"
             and row.get("row") is not None and row.get("col") is not None]
    row_records = [row for row in rows if row.get("level") == "row"
                   and isinstance(row.get("row"), int)]
    if len(cells) < 2:
        return False
    coordinates = [(int(row["row"]), int(row["col"])) for row in cells]
    max_row = max(r for r, _ in coordinates)
    max_col = max(c for _, c in coordinates)
    if operation == "adjacent_row_swap" and max_row >= 1:
        transformed = [(1 if r == 0 else 0 if r == 1 else r, c) for r, c in coordinates]
    elif operation == "adjacent_col_swap" and max_col >= 1:
        transformed = [(r, 1 if c == 0 else 0 if c == 1 else c) for r, c in coordinates]
    else:
        transformed = coordinates[1:] + coordinates[:1]
        operation = "slot_rotation"
    sample_id = cells[0]["sample_id"]
    for cell, (row_index, col_index) in zip(cells, transformed):
        cell["row"], cell["col"] = row_index, col_index
        cell["perturbation"] = operation
        cell["text"] = (f"CELL: [{sample_id}] "
                        f"[perturbed_row_{row_index}_slot_{col_index}] "
                        f"{cell.get('cell_text') or ''}")
    grid = defaultdict(list)
    for cell in sorted(cells, key=lambda row: (int(row["row"]), int(row["col"]),
                                                row["evidence_id"])):
        grid[int(cell["row"])].append(cell.get("cell_text") or "")
    for row_record in row_records:
        row_index = int(row_record["row"])
        values = grid.get(row_index, [])
        row_record["row_cell_texts"] = values
        row_record["perturbation"] = operation
        row_record["text"] = (f"ROW: [{sample_id}] [perturbed_row_{row_index}] "
                              + " ; ".join(values))
    for table in rows:
        if table.get("level") == "table" and table.get("evidence_kind") == "table":
            table["perturbation"] = operation
            table["text"] = (f"TABLE: [{sample_id}] " + " | ".join(
                value for row_index in sorted(grid) for value in grid[row_index]))
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(rate <= 0 or rate > 100 for rate in args.rates):
        raise ValueError("rates must be in (0, 100]")
    original_hash = sha256(args.input)
    original = list(jsonl(args.input))
    by_sample = defaultdict(list)
    for row in original:
        by_sample[row["sample_key"]].append(row)
    sample_keys = sorted(by_sample)
    random.Random(args.seed).shuffle(sample_keys)
    original_multiset = Counter(
        row.get("cell_text") or "" for row in original if row.get("level") == "cell")
    reports = []
    operations = ("adjacent_row_swap", "adjacent_col_swap", "slot_rotation")
    for rate in sorted(args.rates):
        out_dir = args.output_root / f"rate_{rate:02d}"
        output = out_dir / "corpus_test.jsonl"
        if output.exists():
            raise FileExistsError(f"refusing overwrite: {output}")
        selected = set(sample_keys[:round(len(sample_keys) * rate / 100)])
        copied = [dict(row) for row in original]
        copied_by_sample = defaultdict(list)
        for row in copied:
            copied_by_sample[row["sample_key"]].append(row)
        applied, skipped, op_counts = 0, 0, Counter()
        for index, key in enumerate(sample_keys):
            if key not in selected:
                continue
            operation = operations[index % len(operations)]
            if perturb_sample(copied_by_sample[key], operation):
                applied += 1
                op_counts[operation] += 1
            else:
                skipped += 1
        perturbed_multiset = Counter(
            row.get("cell_text") or "" for row in copied if row.get("level") == "cell")
        if perturbed_multiset != original_multiset:
            raise AssertionError(f"rate={rate}: cell-text multiset changed")
        out_dir.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for row in copied:
                handle.write(json.dumps(row) + "\n")
        report = {
            "rate_percent": rate, "seed": args.seed,
            "rate_unit": "fraction of test tables selected",
            "nested_selection": True, "selected_tables": len(selected),
            "total_tables": len(sample_keys), "applied_tables": applied,
            "skipped_tables": skipped, "operation_counts": dict(op_counts),
            "structure_fields_changed_only": ["row", "col", "row_cell_texts", "text"],
            "evidence_ids_unchanged": True, "cell_text_fields_unchanged": True,
            "cell_text_multiset_equal": True,
            "cell_text_count": sum(original_multiset.values()),
            "source_sha256": original_hash, "output_sha256": sha256(output),
        }
        (out_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        logging.info("rate=%d selected=%d applied=%d", rate, len(selected), applied)
    if sha256(args.input) != original_hash:
        raise AssertionError("source corpus changed during perturbation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps({
        "source": str(args.input), "source_sha256_before_after": original_hash,
        "seed": args.seed, "reports": reports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("structure perturbation failed")
        raise
