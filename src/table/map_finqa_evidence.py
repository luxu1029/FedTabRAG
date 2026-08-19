#!/usr/bin/env python3
"""Map FinQA gold_inds to table/row/cell evidence in all variants."""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOG = logging.getLogger("map_finqa_evidence")
VARIANTS = ("flat", "oracle_structure", "predicted_structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finqa-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.35)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = value.replace(",", "")
    value = re.sub(r"\s*%\s*", "%", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ;")


def tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", normalize(text)))


def numeric_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[-+]?\$?\d+(?:\.\d+)?%?", normalize(text)))


def overlap_score(gold: str, candidate: str) -> float:
    left, right = tokens(gold), tokens(candidate)
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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


def best_row(gold: str, rows: Sequence[Dict[str, Any]], threshold: float) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    scored = sorted(((overlap_score(gold, item["text"]), item) for item in rows), key=lambda pair: pair[0], reverse=True)
    score, item = scored[0]
    gold_numbers = numeric_tokens(gold)
    candidate_numbers = numeric_tokens(item["text"])
    if score < threshold or (gold_numbers and not gold_numbers.issubset(candidate_numbers)):
        return None
    return item


def best_row_by_cell_multiset(
    gt_values: Sequence[str], rows: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    wanted = Counter(normalize(value) for value in gt_values if normalize(value))
    if not wanted or not rows:
        return None
    ranked = []
    for item in rows:
        available = Counter(
            normalize(value)
            for value in (item.get("row_cell_texts") or [])
            if normalize(value)
        )
        matched = sum((wanted & available).values())
        ranked.append((matched / sum(wanted.values()), matched, item))
    ratio, matched, item = max(ranked, key=lambda value: (value[0], value[1]))
    return item if ratio >= 0.5 and matched > 0 else None


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    corpora = {variant: read_jsonl(args.corpus_root / variant / "corpus.jsonl") for variant in VARIANTS}
    by_sample: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for variant, items in corpora.items():
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["sample_key"]].append(item)
        by_sample[variant] = grouped

    output = []
    counts = defaultdict(Counter)
    unmapped = []
    for split in ("train", "dev", "test"):
        data = json.load((args.finqa_dir / f"{split}.json").open(encoding="utf-8"))
        for source_index, sample in enumerate(data):
            key = f"{split}:{sample['id']}:{source_index}"
            gold_inds = sample.get("qa", {}).get("gold_inds") or {}
            entry = {
                "query_id": key,
                "sample_key": key,
                "sample_id": sample["id"],
                "split": split,
                "gold_ind_count": len(gold_inds),
                "variants": {},
            }
            for variant in VARIANTS:
                items = by_sample[variant].get(key, [])
                main_tables = [item for item in items if item["evidence_kind"] == "table"]
                contexts = [item for item in items if item["evidence_kind"] == "context"]
                rows = [item for item in items if item["level"] == "row"]
                cells = [item for item in items if item["level"] == "cell"]
                positive = {"table": [], "row": [], "cell": []}
                details = []
                for gold_key, gold_text in gold_inds.items():
                    mapped = {"gold_key": gold_key, "gold_text": gold_text, "levels": {}, "method": None}
                    if gold_key.startswith("text_"):
                        exact = [item for item in contexts if normalize(item.get("cell_text", "")) == normalize(gold_text)]
                        if exact:
                            positive["table"].extend(item["evidence_id"] for item in exact)
                            mapped["levels"]["table"] = [item["evidence_id"] for item in exact]
                            mapped["method"] = "normalized_exact_context"
                        else:
                            mapped["method"] = "unmapped_context"
                    elif gold_key.startswith("table_"):
                        try:
                            gt_row = int(gold_key.split("_", 1)[1])
                        except ValueError:
                            gt_row = -1
                        positive["table"].extend(item["evidence_id"] for item in main_tables)
                        mapped["levels"]["table"] = [item["evidence_id"] for item in main_tables]
                        gt_table = sample.get("table") or []
                        gt_values = (
                            [str(value) if value is not None else "" for value in gt_table[gt_row]]
                            if 0 <= gt_row < len(gt_table)
                            else []
                        )
                        row_item = None
                        if variant == "oracle_structure":
                            row_item = next((item for item in rows if item.get("row") == gt_row), None)
                            mapped["method"] = "gold_row_index"
                        elif variant == "flat":
                            row_item = rows[0] if rows else None
                            mapped["method"] = "flat_sequence"
                        else:
                            row_item = best_row_by_cell_multiset(gt_values, rows)
                            mapped["method"] = "restricted_cell_multiset" if row_item else "unmapped_predicted_row"
                        if row_item:
                            positive["row"].append(row_item["evidence_id"])
                            mapped["levels"]["row"] = [row_item["evidence_id"]]
                        wanted = Counter(normalize(value) for value in gt_values if normalize(value))
                        matched_cells = []
                        remaining = wanted.copy()
                        for item in cells:
                            value = normalize(item.get("cell_text", ""))
                            if value and remaining[value] > 0:
                                matched_cells.append(item["evidence_id"])
                                remaining[value] -= 1
                        positive["cell"].extend(matched_cells)
                        if matched_cells:
                            mapped["levels"]["cell"] = matched_cells
                        mapped["unmatched_gold_cell_values"] = list(remaining.elements())
                    else:
                        mapped["method"] = "unsupported_gold_key"
                    details.append(mapped)
                for level in positive:
                    positive[level] = list(dict.fromkeys(positive[level]))
                    counts[(split, variant)][f"{level}_queries"] += bool(positive[level])
                counts[(split, variant)]["queries"] += 1
                fully_mapped = all(detail.get("method") not in {"unmapped_context", "unmapped_predicted_row", "unsupported_gold_key"} for detail in details)
                if not fully_mapped:
                    unmapped.append({"sample_key": key, "variant": variant, "details": details})
                entry["variants"][variant] = {
                    "positive_evidence_ids": positive,
                    "mapping_details": details,
                    "fully_mapped": fully_mapped,
                }
            output.append(entry)

    atomic_jsonl(args.output, output)
    coverage = {}
    for (split, variant), value in sorted(counts.items()):
        total = value["queries"]
        coverage.setdefault(split, {})[variant] = {
            "queries": total,
            "table_query_coverage": value["table_queries"] / total if total else 0,
            "row_query_coverage": value["row_queries"] / total if total else 0,
            "cell_query_coverage": value["cell_queries"] / total if total else 0,
        }
    report = {
        "seed": args.seed,
        "query_count": len(output),
        "mapping_entry_count": len(output),
        "coverage": coverage,
        "unmapped_variant_records": len(unmapped),
        "unmapped": unmapped,
    }
    atomic_json(args.report, report)
    LOG.info("Mapped queries=%d unmapped_variant_records=%d", len(output), len(unmapped))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; outputs are atomically replaced")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal evidence mapping error")
        sys.exit(2)
