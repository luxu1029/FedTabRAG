"""Auditable reliability/utility weighting for Day-8 federated aggregation."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Iterable

import numpy as np

AGGREGATIONS = ("fedavg", "r_only", "u_only", "r_plus_u")


def jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _average_ranks(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x, y = list(x), list(y)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def client_reliability(partition: dict, selected_clients: list[int],
                       metrics_path: Path, correlation_threshold: float = 0.3) -> dict:
    """Compute R only from each client's local dev structure records."""
    company_owner = partition["company_to_client"]
    records = {cid: [] for cid in selected_clients}
    confidence, s_teds = [], []
    for row in jsonl(metrics_path):
        if row.get("split") != "dev":
            continue
        company = str(row["id"]).split("/", 1)[0]
        if company not in company_owner:
            continue
        cid = int(company_owner[company])
        if cid not in records:
            continue
        records[cid].append(row)
        if math.isfinite(float(row.get("confidence", float("nan")))):
            confidence.append(float(row["confidence"]))
            s_teds.append(float(row.get("s_teds", 0.0)))
    rho = spearman(confidence, s_teds)
    confidence_enabled = abs(rho) >= correlation_threshold
    eta = {"s_teds": 0.4, "valid_html": 0.4,
           "confidence": 0.2 if confidence_enabled else 0.0}
    scale = sum(eta.values())
    eta = {key: value / scale for key, value in eta.items()}
    result = {}
    for cid in selected_clients:
        rows = records[cid]
        if not rows:
            result[str(cid)] = {
                "value": 1.0, "fallback": "empty_local_dev", "dev_tables": 0,
                "components": {}, "eta": eta,
            }
            continue
        components = {
            "s_teds": float(np.mean([float(x.get("s_teds", 0.0)) for x in rows])),
            "valid_html": float(np.mean([bool(x.get("valid_html", False)) for x in rows])),
            "confidence": float(np.mean([float(x.get("confidence", 0.0)) for x in rows])),
        }
        value = sum(eta[key] * components[key] for key in eta)
        result[str(cid)] = {
            "value": float(np.clip(value, 0.0, 1.0)), "fallback": None,
            "dev_tables": len(rows), "components": components, "eta": eta,
        }
    return {
        "clients": result, "confidence_s_teds_spearman": rho,
        "confidence_enabled": confidence_enabled,
        "correlation_threshold": correlation_threshold,
        "source": str(metrics_path), "split": "dev",
    }


def sanitize_utilities(values: list[float]) -> tuple[list[float], list[str | None]]:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    fallback = float(np.mean(finite)) if finite else 1.0
    cleaned, reasons = [], []
    for value in values:
        if math.isfinite(float(value)):
            cleaned.append(float(np.clip(value, 0.0, 1.0)))
            reasons.append(None)
        else:
            cleaned.append(fallback)
            reasons.append("non_finite_or_empty_dev")
    return cleaned, reasons


def quality_values(mode: str, reliability: list[float],
                   utility: list[float]) -> list[float]:
    if mode not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {mode}")
    if len(reliability) != len(utility):
        raise ValueError("R/U length mismatch")
    if mode == "fedavg":
        return [1.0] * len(reliability)
    if mode == "r_only":
        return reliability
    if mode == "u_only":
        return utility
    return [0.5 * r + 0.5 * u for r, u in zip(reliability, utility)]


def _bounded_simplex(raw: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if len(raw) * lower > 1.0 + 1e-12 or len(raw) * upper < 1.0 - 1e-12:
        raise ValueError("infeasible aggregation bounds")
    low_scale, high_scale = 0.0, 1.0
    while np.clip(high_scale * raw, lower, upper).sum() < 1.0:
        high_scale *= 2.0
    for _ in range(100):
        scale = 0.5 * (low_scale + high_scale)
        if np.clip(scale * raw, lower, upper).sum() < 1.0:
            low_scale = scale
        else:
            high_scale = scale
    weights = np.clip(high_scale * raw, lower, upper)
    weights /= weights.sum()
    return weights


def aggregation_weights(sample_counts: list[int], quality: list[float],
                        min_weight: float = 0.05,
                        max_weight: float = 0.5) -> dict:
    counts = np.asarray(sample_counts, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    if len(counts) == 0 or len(counts) != len(q):
        raise ValueError("invalid sample count/quality lengths")
    if (counts <= 0).any() or not np.isfinite(counts).all():
        raise ValueError("sample counts must be positive and finite")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("quality must be non-negative and finite")
    q = np.maximum(q, 1e-8)
    raw = counts * q
    raw_normalized = raw / raw.sum()
    effective_max = max(max_weight, 1.0 / len(counts))
    clipped = _bounded_simplex(raw, min_weight, effective_max)
    if not np.isclose(clipped.sum(), 1.0, atol=1e-12):
        raise AssertionError("aggregation weights do not sum to one")
    return {
        "raw": raw.tolist(), "raw_normalized": raw_normalized.tolist(),
        "clipped": clipped.tolist(), "min_weight": min_weight,
        "max_weight": effective_max,
    }
