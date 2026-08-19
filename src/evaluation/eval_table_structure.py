#!/usr/bin/env python3
"""Evaluate predicted UniTable HTML against FinQA array structure."""

import argparse
import json
import logging
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lxml import etree, html as lxml_html


LOG = logging.getLogger("eval_table_structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        default="s_teds,valid_html,cell_count,row_col_error",
        help="Accepted for protocol compatibility; all listed metrics are computed.",
    )
    parser.add_argument("--unitable-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return values


def html_template(inner: str) -> str:
    return f"<html><body><table>{inner}</table></body></html>"


def gt_html(rows: int, cols: int) -> str:
    row_code = "<tr>" + "<td></td>" * cols + "</tr>"
    return html_template(row_code * rows)


def parse_structure(inner: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    if not inner:
        return False, None, "empty_html"
    try:
        parser = lxml_html.HTMLParser(recover=False, remove_comments=True)
        root = lxml_html.fromstring(html_template(inner), parser=parser)
        tables = root.xpath(".//table")
        if not tables:
            return False, None, "missing_table"
        table = tables[0]
        rows = table.xpath(".//tr")
        cells = table.xpath(".//td")
        if not rows or not cells:
            return False, table, "missing_tr_or_td"
        if any(not row.xpath("./td") for row in rows):
            return False, table, "row_without_td"
        return True, table, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def shape(table: Any) -> Tuple[int, int, int]:
    rows = table.xpath(".//tr")
    row_widths = []
    occupied_cells = 0
    for row in rows:
        width = 0
        for cell in row.xpath("./td"):
            try:
                colspan = max(1, int(cell.attrib.get("colspan", "1")))
                rowspan = max(1, int(cell.attrib.get("rowspan", "1")))
            except ValueError:
                colspan = rowspan = 1
            width += colspan
            occupied_cells += colspan * rowspan
        row_widths.append(width)
    return len(rows), max(row_widths, default=0), occupied_cells


def safe_mean(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    return statistics.fmean(data) if data else None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    unitable_root = str(args.unitable_root.resolve())
    if unitable_root in sys.path:
        sys.path.remove(unitable_root)
    sys.path.insert(0, unitable_root)
    from src.utils.teds import TEDS

    predictions = read_jsonl(args.predictions)
    manifest = read_jsonl(args.manifest)
    pred_by_key = {str(item.get("sample_key")): item for item in predictions}
    teds = TEDS(structure_only=True, n_jobs=1)
    per_sample: List[Dict[str, Any]] = []
    statuses = Counter()

    for truth in manifest:
        key = str(truth.get("sample_key"))
        prediction = pred_by_key.get(key)
        if prediction is None:
            per_sample.append(
                {"sample_key": key, "status": "missing_prediction", "valid_html": False}
            )
            statuses["missing_prediction"] += 1
            continue
        status = str(prediction.get("status"))
        statuses[status] += 1
        valid, table, parse_error = parse_structure(str(prediction.get("html", "")))
        true_rows = int(truth["rows"])
        true_cols = int(truth["cols"])
        true_cells = true_rows * true_cols
        predicted_rows = predicted_cols = predicted_cells = 0
        score = 0.0
        if valid and table is not None and status == "ok":
            predicted_rows, predicted_cols, predicted_cells = shape(table)
            try:
                score = float(
                    teds.evaluate(
                        html_template(str(prediction.get("html", ""))),
                        gt_html(true_rows, true_cols),
                    )
                )
            except Exception as exc:
                valid = False
                parse_error = f"TEDS_{type(exc).__name__}: {exc}"
                score = 0.0
        cell_ratio = predicted_cells / true_cells if true_cells else None
        per_sample.append(
            {
                "sample_key": key,
                "id": truth.get("id"),
                "split": truth.get("split"),
                "status": status,
                "valid_html": valid,
                "parse_error": parse_error,
                "s_teds": score,
                "confidence": prediction.get("confidence"),
                "true_rows": true_rows,
                "true_cols": true_cols,
                "true_cells": true_cells,
                "predicted_rows": predicted_rows,
                "predicted_cols": predicted_cols,
                "predicted_cells": predicted_cells,
                "cell_count_ratio": cell_ratio,
                "cell_count_exact": predicted_cells == true_cells,
                "row_absolute_error": abs(predicted_rows - true_rows),
                "col_absolute_error": abs(predicted_cols - true_cols),
                "elapsed_seconds": prediction.get("elapsed_seconds"),
                "ended_with_eos": prediction.get("ended_with_eos"),
            }
        )

    count = len(manifest)
    valid_items = [item for item in per_sample if item.get("valid_html")]
    evaluated_items = [item for item in per_sample if "s_teds" in item]
    by_split: Dict[str, Dict[str, Any]] = {}
    for split in sorted({str(item.get("split")) for item in per_sample}):
        items = [item for item in per_sample if str(item.get("split")) == split]
        by_split[split] = {
            "count": len(items),
            "valid_html_rate": (
                sum(bool(item.get("valid_html")) for item in items) / len(items)
                if items
                else None
            ),
            "mean_s_teds": safe_mean(float(item.get("s_teds", 0.0)) for item in items),
            "cell_count_exact_rate": (
                sum(bool(item.get("cell_count_exact")) for item in items) / len(items)
                if items
                else None
            ),
        }

    report = {
        "predictions": str(args.predictions),
        "manifest": str(args.manifest),
        "count": count,
        "prediction_count": len(predictions),
        "status_counts": dict(statuses),
        "missing_prediction_count": sum(
            item.get("status") == "missing_prediction" for item in per_sample
        ),
        "valid_html_count": len(valid_items),
        "valid_html_rate": len(valid_items) / count if count else None,
        "mean_s_teds": safe_mean(float(item.get("s_teds", 0.0)) for item in per_sample),
        "cell_count": {
            "exact_count": sum(bool(item.get("cell_count_exact")) for item in per_sample),
            "exact_rate": (
                sum(bool(item.get("cell_count_exact")) for item in per_sample) / count
                if count
                else None
            ),
            "mean_ratio": safe_mean(
                float(item["cell_count_ratio"])
                for item in per_sample
                if item.get("cell_count_ratio") is not None
            ),
            "mean_absolute_error": safe_mean(
                abs(int(item.get("predicted_cells", 0)) - int(item.get("true_cells", 0)))
                for item in per_sample
                if "true_cells" in item
            ),
        },
        "row_col_error": {
            "mean_row_absolute_error": safe_mean(
                float(item["row_absolute_error"])
                for item in per_sample
                if "row_absolute_error" in item
            ),
            "mean_col_absolute_error": safe_mean(
                float(item["col_absolute_error"])
                for item in per_sample
                if "col_absolute_error" in item
            ),
        },
        "inference": {
            "mean_elapsed_seconds": safe_mean(
                float(item["elapsed_seconds"])
                for item in per_sample
                if item.get("elapsed_seconds") is not None
            ),
            "eos_rate": (
                sum(bool(item.get("ended_with_eos")) for item in per_sample) / count
                if count
                else None
            ),
        },
        "by_split": by_split,
        "invalid_samples": [
            {
                "sample_key": item.get("sample_key"),
                "status": item.get("status"),
                "reason": item.get("parse_error"),
            }
            for item in per_sample
            if not item.get("valid_html")
        ],
    }
    atomic_json(args.output, report)
    per_sample_path = args.per_sample_output or args.output.with_name(
        args.output.stem + "_per_sample.jsonl"
    )
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = per_sample_path.with_suffix(per_sample_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in per_sample:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(per_sample_path)
    LOG.info(
        "Evaluated count=%d valid_html=%d mean_s_teds=%s output=%s",
        count,
        len(valid_items),
        report["mean_s_teds"],
        args.output,
    )
    return 1 if report["missing_prediction_count"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal structure evaluation error")
        sys.exit(2)
