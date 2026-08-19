"""In-process exact dev retrieval used only for checkpoint selection."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

LEVELS = ("table", "row", "cell")


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class DevEvaluator:
    def __init__(self, partition, variant="predicted_structure", selected_clients=None):
        root = Path("data/processed") / variant
        selected = set(selected_clients if selected_clients is not None
                       else range(partition["num_clients"]))
        company_owner = partition["company_to_client"]
        self.corpus = [x for x in _jsonl(root / "corpus_dev.jsonl")
                       if int(company_owner[x["company"]]) in selected]
        self.queries = [x for x in _jsonl(root / "queries_dev.jsonl")
                        if int(company_owner[x["company"]]) in selected]
        self.maps = {x["query_id"]: x for x in _jsonl("data/processed/evidence_map.jsonl")
                     if x["split"] == "dev"}
        self.variant = variant
        self.selected = sorted(selected)
        self.owner = {qid: int(cid) for cid, data in partition["clients"].items()
                      for qid in data["query_ids"]["dev"]}
        self.evidence_index = {x["evidence_id"]: i for i, x in enumerate(self.corpus)}

    @staticmethod
    def _encode(model, tokenizer, texts, cfg, device):
        out = []
        model.eval()
        for start in range(0, len(texts), 128):
            x = tokenizer(texts[start:start + 128], padding=True, truncation=True,
                          max_length=cfg["max_length"], return_tensors="pt")
            x = {k: v.to(device) for k, v in x.items()}
            with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda"):
                out.append(model(x).float().cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def _ndcg(ranks):
        if not ranks:
            return 0.0
        dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= 10)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(ranks), 10) + 1))
        return dcg / ideal

    @staticmethod
    def _map(ranks):
        ranks = sorted(ranks)
        return (sum(i / rank for i, rank in enumerate(ranks, 1)) / len(ranks)
                if ranks else 0.0)

    def evaluate(self, model, tokenizer, cfg, device):
        corpus_emb = self._encode(model, tokenizer, [x["text"] for x in self.corpus],
                                  cfg, device)
        query_emb = self._encode(model, tokenizer, [x["question"] for x in self.queries],
                                 cfg, device)
        per_level = {}
        clients = {}
        for level in LEVELS:
            indices = np.asarray([i for i, x in enumerate(self.corpus) if x["level"] == level])
            reverse = {int(g): i for i, g in enumerate(indices)}
            values = []
            by_client = {i: [] for i in self.selected}
            for qs in range(0, len(self.queries), 64):
                scores = query_emb[qs:qs + 64] @ corpus_emb[indices].T
                for off, query in enumerate(self.queries[qs:qs + len(scores)]):
                    ids = self.maps[query["query_id"]]["variants"][self.variant][
                        "positive_evidence_ids"][level]
                    positive = [reverse[self.evidence_index[e]] for e in ids
                                if e in self.evidence_index and self.evidence_index[e] in reverse]
                    if not positive:
                        continue
                    row = scores[off]
                    ranks = [int(np.count_nonzero(row > row[p]) +
                                 np.count_nonzero((row == row[p]) &
                                                  (np.arange(len(row)) < p)) + 1)
                             for p in positive]
                    item = {
                        "recall@1": float(min(ranks) <= 1),
                        "recall@5": float(min(ranks) <= 5),
                        "mrr": 1.0 / min(ranks),
                        "map": self._map(ranks),
                        "ndcg@10": self._ndcg(ranks),
                    }
                    values.append(item)
                    by_client[self.owner[query["query_id"]]].append(item)
            mean = lambda xs, key: statistics.fmean(x[key] for x in xs) if xs else 0.0
            per_level[level] = {"eligible_queries": len(values),
                                **{k: mean(values, k) for k in values[0]}}
            client_metric = {str(cid): {k: mean(rows, k) for k in
                                       ("recall@1", "recall@5", "mrr", "map", "ndcg@10")}
                             for cid, rows in by_client.items()}
            clients[level] = client_metric
            vals = [x["ndcg@10"] for x in client_metric.values()]
            per_level[level]["macro_ndcg@10"] = statistics.fmean(vals)
            per_level[level]["worst_ndcg@10"] = min(vals)
            per_level[level]["std_ndcg@10"] = statistics.pstdev(vals)
        score = (0.2 * per_level["table"]["ndcg@10"] +
                 0.3 * per_level["row"]["ndcg@10"] +
                 0.5 * per_level["cell"]["ndcg@10"])
        model.train()
        return {"weighted_ndcg": score, "levels": per_level, "clients": clients}
