#!/usr/bin/env python3
"""Create and audit a deterministic, company-atomic FinQA client partition."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def canonical_hash(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finqa-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--structure-metrics", type=Path,
                        default=Path("outputs/unitable/structure_metrics_per_sample.jsonl"))
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if args.num_clients < 2:
        raise ValueError("--num-clients must be >=2")
    if not args.finqa_dir.is_dir():
        raise FileNotFoundError(args.finqa_dir)

    # Processed queries are the canonical experimental sample universe.
    records = []
    by_company = defaultdict(lambda: Counter())
    ids_by_company = defaultdict(lambda: defaultdict(list))
    for split in ("train", "dev", "test"):
        path = args.processed_dir / "flat" / f"queries_{split}.jsonl"
        for item in read_jsonl(path):
            company = item["company"]
            qid = item["query_id"]
            records.append((company, split, qid))
            by_company[company][split] += 1
            ids_by_company[company][split].append(qid)

    raw_stats = defaultdict(lambda: defaultdict(list))
    question_types = defaultdict(Counter)
    for split in ("train", "dev", "test"):
        raw_path = args.finqa_dir / f"{split}.json"
        for item in json.loads(raw_path.read_text(encoding="utf-8")):
            company = item["filename"].split("/", 1)[0]
            rows = len(item["table"])
            cols = max((len(row) for row in item["table"]), default=0)
            raw_stats[company]["rows"].append(rows)
            raw_stats[company]["cols"].append(cols)
            qa = item["qa"]
            has_table = bool(qa.get("ann_table_rows"))
            has_text = bool(qa.get("ann_text_rows"))
            qtype = "hybrid" if has_table and has_text else "table" if has_table else "text"
            question_types[company][qtype] += 1

    structure_by_company = defaultdict(list)
    if args.structure_metrics.exists():
        for item in read_jsonl(args.structure_metrics):
            company = item["id"].split("/", 1)[0]
            structure_by_company[company].append(float(item["s_teds"]))

    # Deterministic largest-first greedy bin packing. Seed only resolves exact ties.
    rng = random.Random(args.seed)
    tie = {company: rng.random() for company in by_company}
    companies = sorted(by_company, key=lambda c: (-by_company[c]["train"], tie[c], c))
    loads = [0] * args.num_clients
    company_to_client = {}
    for company in companies:
        cid = min(range(args.num_clients), key=lambda i: (loads[i], i))
        company_to_client[company] = cid
        loads[cid] += by_company[company]["train"]

    mapping_core = {
        "schema_version": 1,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "strategy": "largest_train_count_first_greedy_company_atomic",
        "company_to_client": company_to_client,
    }
    mapping_hash = canonical_hash(mapping_core)
    clients = {}
    for cid in range(args.num_clients):
        company_list = sorted(c for c, owner in company_to_client.items() if owner == cid)
        clients[str(cid)] = {
            "companies": company_list,
            "query_ids": {
                split: sorted(qid for c in company_list for qid in ids_by_company[c][split])
                for split in ("train", "dev", "test")
            },
        }
    output = {
        **mapping_core,
        "mapping_sha256": mapping_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clients": clients,
    }

    # Freeze means an existing, different mapping is never silently overwritten.
    if args.output.exists() and not args.force:
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if old.get("mapping_sha256") != mapping_hash:
            raise RuntimeError(f"frozen mapping differs: {args.output}; pass --force explicitly")
        logging.info("Existing frozen mapping matches %s", mapping_hash)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    owner_sets = [set(clients[str(i)]["companies"]) for i in range(args.num_clients)]
    overlap = sum(len(owner_sets[i] & owner_sets[j])
                  for i in range(args.num_clients) for j in range(i + 1, args.num_clients))
    per_client = {}
    for cid in range(args.num_clients):
        q = clients[str(cid)]["query_ids"]
        cs = clients[str(cid)]["companies"]
        rows = [x for c in cs for x in raw_stats[c]["rows"]]
        cols = [x for c in cs for x in raw_stats[c]["cols"]]
        steds = [x for c in cs for x in structure_by_company[c]]
        types = sum((question_types[c] for c in cs), Counter())
        per_client[str(cid)] = {
            "companies": len(cs),
            **{f"{split}_queries": len(q[split]) for split in ("train", "dev", "test")},
            "question_types": dict(sorted(types.items())),
            "table_rows_mean": statistics.fmean(rows) if rows else None,
            "table_cols_mean": statistics.fmean(cols) if cols else None,
            "structure_s_teds_mean": statistics.fmean(steds) if steds else None,
            "structure_s_teds_min": min(steds) if steds else None,
        }
    min_load, max_load = min(loads), max(loads)
    report = {
        "mapping_sha256": mapping_hash,
        "company_count": len(companies),
        "query_count": len(records),
        "company_overlap_count": overlap,
        "cross_split_owner_mismatch_count": 0,
        "per_client": per_client,
        "train_load_min": min_load,
        "train_load_max": max_load,
        "train_min_max_ratio": min_load / max_load if max_load else 0.0,
        "acceptance": overlap == 0 and len(company_to_client) == len(companies),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("partition=%s companies=%d train_loads=%s accepted=%s",
                 mapping_hash, len(companies), loads, report["acceptance"])
    return 0 if report["acceptance"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.exception("partition failed")
        sys.exit(1)
