from src.retrieval.evaluate_confidence_gating import (
    COMPOSITE_WEIGHTS,
    normalized_consistency,
    reliability,
    use_unitable,
)


def structure(**overrides):
    row = {
        "s_teds": 0.8,
        "cell_count_exact": False,
        "predicted_cells": 8,
        "true_cells": 10,
        "valid_html": True,
        "confidence": 0.9,
    }
    row.update(overrides)
    return row


def test_reliability_formula_and_consistency():
    row = structure()
    assert normalized_consistency(row) == 0.8
    expected = (
        COMPOSITE_WEIGHTS["s_teds"] * 0.8
        + COMPOSITE_WEIGHTS["cell_count_consistency"] * 0.8
        + COMPOSITE_WEIGHTS["valid_html"]
        + COMPOSITE_WEIGHTS["confidence"] * 0.9
    )
    assert abs(reliability(row) - expected) < 1e-12
    assert normalized_consistency(structure(cell_count_exact=True)) == 1.0


def test_gate_requires_all_configured_filters():
    row = structure()
    config = {
        "s_teds_threshold": 0.75,
        "cell_count_consistency_threshold": 0.75,
        "require_valid_html": True,
    }
    assert use_unitable(row, config)
    assert not use_unitable(structure(valid_html=False), config)
    assert not use_unitable(structure(s_teds=0.70), config)


def test_force_baselines_and_empty_gate_are_safe():
    row = structure()
    assert not use_unitable(row, {"force": "flat"})
    assert use_unitable(row, {"force": "unitable"})
    assert not use_unitable(row, {})
