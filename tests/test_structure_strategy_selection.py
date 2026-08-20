from src.evaluation.select_and_freeze_structure_strategy import THRESHOLDS, judge


BASELINE = {
    "weighted_ndcg@10": 0.20,
    "row_recall@5": 0.30,
    "cell_recall@5": 0.40,
    "worst_client_weighted_ndcg@10": 0.10,
}


def candidate(ndcg=0.203, row=0.31, cell=0.40, worst=0.095):
    return {
        "weighted_ndcg@10": ndcg,
        "row_recall@5": row,
        "cell_recall@5": cell,
        "worst_client_weighted_ndcg@10": worst,
    }


def test_thresholds_are_the_preregistered_values():
    assert THRESHOLDS == {
        "weighted_ndcg_min_improvement": 0.003,
        "row_or_cell_recall5_min_improvement": 0.01,
        "worst_client_max_degradation": 0.005,
    }


def test_candidate_at_thresholds_passes_with_float_tolerance_inputs():
    # Values slightly above exact decimal thresholds avoid binary boundary noise.
    result = judge(candidate(ndcg=0.2030000001, row=0.3100000001,
                             worst=0.0950000001), BASELINE)
    assert result["passes"]
    assert all(result["checks"].values())


def test_each_gate_is_mandatory():
    assert not judge(candidate(ndcg=0.2029), BASELINE)["passes"]
    assert not judge(candidate(row=0.309, cell=0.409), BASELINE)["passes"]
    assert not judge(candidate(worst=0.0949), BASELINE)["passes"]


def test_none_never_passes_improvement_gate():
    result = judge(BASELINE, BASELINE)
    assert not result["passes"]
    assert result["deltas_vs_canonical_flat"]["weighted_ndcg@10"] == 0.0
