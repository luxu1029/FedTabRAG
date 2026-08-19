#!/usr/bin/env python3
"""Build Flat, oracle-grid, and UniTable-predicted hierarchical evidence."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from lxml import html as lxml_html


LOG = logging.getLogger("build_hierarchical_evidence")
VARIANTS = ("flat", "oracle_structure", "predicted_structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finqa-dir", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--unitable-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/data/evidence_schema.json"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
    return output


def atomic_jsonl(path: Path, values: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_finqa(path: Path, split: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a list")
    output = []
    for index, item in enumerate(values):
        copy = dict(item)
        copy["_split"] = split
        copy["_source_index"] = index
        copy["_sample_key"] = f"{split}:{item.get('id')}:{index}"
        output.append(copy)
    return output


def evidence(
    sample: Dict[str, Any],
    variant: str,
    level: str,
    kind: str,
    index: str,
    text: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    flat_index: Optional[int] = None,
    structure_valid: bool = True,
    overflow: bool = False,
    cell_text: Optional[str] = None,
    row_cell_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sample_key = sample["_sample_key"]
    return {
        "evidence_id": f"{sample_key}::{level}::{index}",
        "sample_key": sample_key,
        "sample_id": str(sample.get("id") or ""),
        "split": sample["_split"],
        "company": str(sample.get("filename") or sample.get("id") or "").split("/", 1)[0],
        "level": level,
        "evidence_kind": kind,
        "text": text,
        "row": row,
        "col": col,
        "flat_index": flat_index,
        "source_variant": variant,
        "structure_valid": structure_valid,
        "overflow": overflow,
        "cell_text": cell_text,
        "row_cell_texts": row_cell_texts,
    }


def context_evidence(sample: Dict[str, Any], variant: str) -> List[Dict[str, Any]]:
    output = []
    paragraphs = list(sample.get("pre_text") or []) + list(sample.get("post_text") or [])
    for index, paragraph in enumerate(paragraphs):
        content = str(paragraph)
        output.append(
            evidence(
                sample, variant, "table", "context", f"context_{index}",
                f"TABLE: [{sample['id']}] [context_{index}] {content}",
                cell_text=content,
            )
        )
    return output


def table_text(sample_id: str, cells: Sequence[str]) -> str:
    return f"TABLE: [{sample_id}] " + " | ".join(cells)


def row_text(sample_id: str, label: str, cells: Sequence[str]) -> str:
    return f"ROW: [{sample_id}] [{label}] " + " ; ".join(cells)


def cell_text(sample_id: str, label: str, value: str) -> str:
    return f"CELL: [{sample_id}] [{label}] {value}"


def flat_grid(sample: Dict[str, Any], cells: List[str]) -> List[Dict[str, Any]]:
    sample_id = sample["id"]
    output = [
        evidence(sample, "flat", "table", "table", "main", table_text(sample_id, cells)),
        evidence(sample, "flat", "row", "flat_sequence", "flat_0",
                 row_text(sample_id, "flat_sequence", cells), row=0,
                 row_cell_texts=list(cells)),
    ]
    for index, value in enumerate(cells):
        output.append(
            evidence(
                sample, "flat", "cell", "cell", str(index),
                cell_text(sample_id, f"flat_{index}", value),
                row=None, col=None, flat_index=index, cell_text=value,
            )
        )
    return output


def oracle_grid(sample: Dict[str, Any], cells: List[str]) -> List[Dict[str, Any]]:
    table = sample["table"]
    sample_id = sample["id"]
    output = [
        evidence(sample, "oracle_structure", "table", "table", "main",
                 table_text(sample_id, cells))
    ]
    flat_index = 0
    for row_index, row in enumerate(table):
        row_cells = [str(value) if value is not None else "" for value in row]
        output.append(
            evidence(
                sample, "oracle_structure", "row", "row", str(row_index),
                row_text(sample_id, f"row_{row_index}", row_cells), row=row_index,
                row_cell_texts=row_cells,
            )
        )
        for col_index, value in enumerate(row_cells):
            output.append(
                evidence(
                    sample, "oracle_structure", "cell", "cell",
                    f"{row_index}_{col_index}",
                    cell_text(sample_id, f"row_{row_index}_col_{col_index}", value),
                    row=row_index, col=col_index, flat_index=flat_index, cell_text=value,
                )
            )
            flat_index += 1
    return output


def predicted_slots(inner_html: str) -> List[List[Dict[str, int]]]:
    root = lxml_html.fromstring(f"<table>{inner_html}</table>")
    tables = root.xpath("self::table|.//table")
    if not tables:
        raise ValueError("prediction has no table")
    slots = []
    for row in tables[0].xpath(".//tr"):
        row_slots = []
        for cell in row.xpath("./td"):
            try:
                colspan = max(1, int(cell.attrib.get("colspan", "1")))
                rowspan = max(1, int(cell.attrib.get("rowspan", "1")))
            except ValueError as exc:
                raise ValueError(f"invalid span: {exc}") from exc
            row_slots.append({"colspan": colspan, "rowspan": rowspan})
        if row_slots:
            slots.append(row_slots)
    if not slots:
        raise ValueError("prediction has no row/cell slots")
    return slots


def predicted_grid(
    sample: Dict[str, Any], cells: List[str], prediction: Dict[str, Any]
) -> List[Dict[str, Any]]:
    sample_id = sample["id"]
    valid = prediction.get("status") == "ok"
    slots = predicted_slots(str(prediction.get("html", ""))) if valid else []
    assigned: List[List[Dict[str, Any]]] = []
    position = 0
    for row in slots:
        assigned_row = []
        for slot in row:
            value = cells[position] if position < len(cells) else ""
            assigned_row.append({**slot, "value": value, "flat_index": position})
            position += 1
        assigned.append(assigned_row)

    output = [
        evidence(
            sample, "predicted_structure", "table", "table", "main",
            table_text(sample_id, cells), structure_valid=valid,
        )
    ]
    for row_index, row in enumerate(assigned):
        values = [slot["value"] for slot in row]
        output.append(
            evidence(
                sample, "predicted_structure", "row", "row", str(row_index),
                row_text(sample_id, f"predicted_row_{row_index}", values),
                row=row_index, structure_valid=valid, row_cell_texts=values,
            )
        )
        for col_index, slot in enumerate(row):
            value = slot["value"]
            output.append(
                {
                    **evidence(
                        sample, "predicted_structure", "cell", "cell",
                        f"{row_index}_{col_index}",
                        cell_text(sample_id, f"predicted_row_{row_index}_slot_{col_index}", value),
                        row=row_index, col=col_index, flat_index=slot["flat_index"],
                        structure_valid=valid, cell_text=value,
                    ),
                    "colspan": slot["colspan"],
                    "rowspan": slot["rowspan"],
                }
            )
    for overflow_index, value in enumerate(cells[position:]):
        flat_index = position + overflow_index
        output.append(
            evidence(
                sample, "predicted_structure", "cell", "overflow", f"overflow_{overflow_index}",
                cell_text(sample_id, f"overflow_{overflow_index}", value),
                flat_index=flat_index, structure_valid=valid, overflow=True, cell_text=value,
            )
        )
    return output


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if any(value not in VARIANTS for value in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    manifest = {item["sample_key"]: item for item in read_jsonl(args.render_manifest)}
    predictions = {item["sample_key"]: item for item in read_jsonl(args.unitable_predictions)}
    samples = []
    for split in ("train", "dev", "test"):
        samples.extend(load_finqa(args.finqa_dir / f"{split}.json", split))
    if len(samples) != len(manifest):
        raise ValueError(f"FinQA/manifest count mismatch: {len(samples)} vs {len(manifest)}")

    corpora: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    queries: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    stats: Dict[str, Dict[str, int]] = {
        variant: {"tables": 0, "rows": 0, "cells": 0, "overflow_cells": 0}
        for variant in variants
    }
    for number, sample in enumerate(samples, 1):
        key = sample["_sample_key"]
        source = manifest.get(key)
        if source is None:
            raise KeyError(f"Missing manifest entry {key}")
        cells = [str(value) for value in source["cell_text_row_major"]]
        for variant in variants:
            if variant == "flat":
                values = flat_grid(sample, cells)
            elif variant == "oracle_structure":
                values = oracle_grid(sample, cells)
            else:
                prediction = predictions.get(key)
                if prediction is None:
                    raise KeyError(f"Missing prediction {key}")
                values = predicted_grid(sample, cells, prediction)
            values.extend(context_evidence(sample, variant))
            corpora[variant].extend(values)
            queries[variant].append(
                {
                    "query_id": key,
                    "sample_key": key,
                    "sample_id": sample["id"],
                    "split": sample["_split"],
                    "company": str(sample.get("filename") or sample["id"]).split("/", 1)[0],
                    "question": str(sample.get("qa", {}).get("question") or ""),
                    "source_variant": variant,
                }
            )
            stats[variant]["tables"] += sum(item["level"] == "table" for item in values)
            stats[variant]["rows"] += sum(item["level"] == "row" for item in values)
            stats[variant]["cells"] += sum(item["level"] == "cell" for item in values)
            stats[variant]["overflow_cells"] += sum(bool(item["overflow"]) for item in values)
        if number % 1000 == 0:
            LOG.info("Prepared %d/%d samples", number, len(samples))

    for variant in variants:
        root = args.output_root / variant
        atomic_jsonl(root / "corpus.jsonl", corpora[variant])
        for split in ("train", "dev", "test"):
            atomic_jsonl(
                root / f"corpus_{split}.jsonl",
                (item for item in corpora[variant] if item["split"] == split),
            )
            atomic_jsonl(
                root / f"queries_{split}.jsonl",
                (item for item in queries[variant] if item["split"] == split),
            )
        atomic_json(
            root / "metadata.json",
            {
                "schema_version": schema["schema_version"],
                "variant": variant,
                "seed": args.seed,
                "record_counts": stats[variant],
                "render_manifest": str(args.render_manifest),
                "unitable_predictions": str(args.unitable_predictions) if variant == "predicted_structure" else None,
            },
        )
        LOG.info("Wrote %s records=%d", variant, len(corpora[variant]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; rerun the command (outputs are atomically replaced)")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal evidence build error")
        sys.exit(2)
