import math

import pytest

from src.federated.aggregation_ru import (
    aggregation_weights,
    quality_values,
    sanitize_utilities,
)


def test_equal_quality_reduces_to_sample_weighting():
    result = aggregation_weights([10, 20, 30], [0.5, 0.5, 0.5],
                                 min_weight=0.0, max_weight=1.0)
    assert result["clipped"] == pytest.approx([1 / 6, 2 / 6, 3 / 6])


def test_higher_quality_increases_weight_without_exceeding_cap():
    base = aggregation_weights([10, 10, 10], [0.5, 0.5, 0.5],
                               min_weight=0.05, max_weight=0.5)
    raised = aggregation_weights([10, 10, 10], [0.9, 0.5, 0.5],
                                 min_weight=0.05, max_weight=0.5)
    assert raised["clipped"][0] > base["clipped"][0]
    assert raised["clipped"][0] <= 0.5


def test_nan_utility_has_explicit_fallback():
    cleaned, reasons = sanitize_utilities([0.2, math.nan, 0.4])
    assert cleaned == pytest.approx([0.2, 0.3, 0.4])
    assert reasons == [None, "non_finite_or_empty_dev", None]


def test_clipped_weights_are_finite_nonnegative_and_sum_to_one():
    result = aggregation_weights([1, 100, 2], [0.01, 1.0, 0.2],
                                 min_weight=0.05, max_weight=0.5)
    assert sum(result["clipped"]) == pytest.approx(1.0, abs=1e-12)
    assert min(result["clipped"]) >= 0.05
    assert max(result["clipped"]) <= 0.5
    assert all(math.isfinite(x) for x in result["clipped"])


def test_four_modes_only_change_quality_definition():
    r, u = [0.8, 0.4], [0.2, 0.6]
    assert quality_values("fedavg", r, u) == [1.0, 1.0]
    assert quality_values("r_only", r, u) == r
    assert quality_values("u_only", r, u) == u
    assert quality_values("r_plus_u", r, u) == pytest.approx([0.5, 0.5])
