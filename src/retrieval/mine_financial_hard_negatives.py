#!/usr/bin/env python3
"""Mine deterministic financial hard negatives within each frozen train client."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r"[a-z]+|\d{4}|[-+]?\d+(?:\.\d+)?%?")
STOP = {"the", "a", "an", "of", "to", "in", "and", "is", "was", "were", "what",
        "how", "much", "for", "on", "by", "from"}


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def tokens(text):
    return {x for x in TOKEN.findall(text.lower()) if x not in STOP and len(x) > 1}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--evidence-map", type=Path, default=Path("data/processed/evidence_map.jsonl"))
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--types", default="same_table_wrong_row,same_row_wrong_cell,same_metric_wrong_period")
    p.add_argument("--per-query", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sample-output", type=Path,
                   default=Path("outputs/audit/hard_negative_sample_100.jsonl"))
    return p.parse_args()


def main():
    args = parse_args(); random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    requested = set(args.types.split(","))
    partition = json.loads(args.partition.read_text())
    company_client = partition["company_to_client"]
    queries = {x["query_id"]: x for x in jsonl(args.queries)}
    positives = {}
    for x in jsonl(args.evidence_map):
        if x["split"] == "train" and x["query_id"] in queries:
            p = x["variants"]["predicted_structure"]["positive_evidence_ids"]
            positives[x["query_id"]] = {e for level in p.values() for e in level}

    evidence = {}
    sample_level = defaultdict(lambda: defaultdict(list))
    row_cells = defaultdict(list)
    token_index = defaultdict(lambda: defaultdict(list))
    for x in jsonl(args.corpus):
        eid = x["evidence_id"]; evidence[eid] = x
        sample_level[x["sample_key"]][x["level"]].append(eid)
        if x["level"] == "cell" and x.get("row") is not None:
            row_cells[(x["sample_key"], x["row"])].append(eid)
        cid = int(company_client[x["company"]])
        if x["level"] in ("row", "table"):
            for tok in sorted(tokens(x["text"]))[:24]:
                if len(token_index[cid][tok]) < 500:
                    token_index[cid][tok].append(eid)

    rows, type_counts = [], Counter()
    local_violations = split_violations = positive_violations = 0
    for qid in sorted(queries):
        q = queries[qid]; cid = int(company_client[q["company"]])
        pos = positives.get(qid, set()); candidates = []
        sample = q["sample_key"]
        if "same_table_wrong_row" in requested:
            for eid in sample_level[sample]["row"]:
                if eid not in pos:
                    candidates.append(("same_table_wrong_row", eid))
        if "same_row_wrong_cell" in requested:
            positive_rows = {evidence[e]["row"] for e in pos
                             if e in evidence and evidence[e]["level"] in ("row", "cell")
                             and evidence[e].get("row") is not None}
            cell_pool = [e for row in positive_rows for e in row_cells[(sample, row)]]
            for eid in cell_pool:
                if eid not in pos:
                    candidates.append(("same_row_wrong_cell", eid))
        if "same_metric_wrong_period" in requested:
            lexical = Counter()
            for tok in tokens(q["question"]):
                for eid in token_index[cid].get(tok, []):
                    x = evidence[eid]
                    q_parts = q["sample_id"].split("/")
                    x_parts = x["sample_id"].split("/")
                    wrong_entity_or_period = (x["company"] != q["company"]
                                              or (len(q_parts) > 1 and len(x_parts) > 1
                                                  and x_parts[1] != q_parts[1]))
                    if x["sample_key"] != sample and wrong_entity_or_period and eid not in pos:
                        lexical[eid] += 1
            for eid, _ in sorted(lexical.items(), key=lambda z: (-z[1], z[0]))[:20]:
                candidates.append(("same_metric_wrong_period", eid))
        # Stable type-round-robin selection maximizes type coverage.
        grouped = defaultdict(list)
        for kind, eid in candidates:
            if eid not in grouped[kind]: grouped[kind].append(eid)
        chosen = []
        order = sorted(requested)
        cursor = 0
        while len(chosen) < args.per_query and any(grouped.values()):
            kind = order[cursor % len(order)]; cursor += 1
            if grouped[kind]:
                eid = grouped[kind].pop(0)
                if eid not in {x["evidence_id"] for x in chosen}:
                    chosen.append({"type": kind, "evidence_id": eid})
                    type_counts[kind] += 1
        for item in chosen:
            x = evidence[item["evidence_id"]]
            item.update({"text": x["text"], "level": x["level"],
                         "sample_key": x["sample_key"], "company": x["company"]})
            local_violations += int(company_client[x["company"]] != cid)
            split_violations += int(x["split"] != "train")
            positive_violations += int(x["evidence_id"] in pos)
        rows.append({"query_id": qid, "client_id": cid, "split": "train",
                     "negatives": chosen})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    tmp.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    flat = [{"query_id": row["query_id"], "question": queries[row["query_id"]]["question"],
             "client_id": row["client_id"], **neg}
            for row in rows for neg in row["negatives"]]
    sample = random.Random(args.seed).sample(flat, min(100, len(flat)))
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_output.open("w") as f:
        for item in sample: f.write(json.dumps(item) + "\n")
    report = {
        "query_count": len(rows), "queries_with_negative": sum(bool(x["negatives"]) for x in rows),
        "negative_count": sum(len(x["negatives"]) for x in rows),
        "type_counts": dict(type_counts), "per_query": args.per_query,
        "source_split": "train", "cross_client_violation_count": local_violations,
        "non_train_violation_count": split_violations,
        "positive_as_negative_count": positive_violations,
        "cache_sha256": digest,
        "manual_sample_path": str(args.sample_output),
        "manual_sample_count": len(sample),
        "accepted": not (local_violations or split_violations or positive_violations),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("%s", report)


if __name__ == "__main__":
    try: main()
    except Exception:
        logging.exception("mining failed"); sys.exit(1)
