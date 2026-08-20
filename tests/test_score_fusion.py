from src.retrieval.evaluate_score_fusion import fuse, normalize, ranking_metrics


def candidates(prefix, scores):
    return [
        {"evidence_id": f"{prefix}{index}", "score": score}
        for index, score in enumerate(scores)
    ]


def test_normalizations_and_missing_score_are_deterministic():
    rows = candidates("a", [3.0, 2.0, 1.0])
    minmax, missing = normalize(rows, "min-max")
    assert list(minmax.values()) == [1.0, 0.5, 0.0]
    assert missing == -1.0
    ranks, missing = normalize(rows, "rank_score")
    assert list(ranks.values()) == [1.0, 0.5, 0.0]
    assert missing == -1.0
    zscores, missing = normalize(rows, "z-score")
    assert abs(sum(zscores.values())) < 1e-12
    assert missing < min(zscores.values())


def test_alpha_endpoints_preserve_the_selected_branch_order():
    flat = candidates("f", [0.9, 0.8, 0.7])
    unitable = candidates("u", [0.95, 0.85, 0.75])
    for method in ("z-score", "min-max", "rank_score"):
        assert [item[0] for item in fuse(flat, unitable, method, 1.0)[:3]] == [
            "f0", "f1", "f2"]
        assert [item[0] for item in fuse(flat, unitable, method, 0.0)[:3]] == [
            "u0", "u1", "u2"]


def test_exact_id_alignment_combines_shared_candidate_scores():
    flat = [{"evidence_id": "shared", "score": 0.9},
            {"evidence_id": "flat", "score": 0.1}]
    unitable = [{"evidence_id": "shared", "score": 0.8},
                {"evidence_id": "unitable", "score": 0.2}]
    ranking = fuse(flat, unitable, "min-max", 0.5)
    assert ranking[0][0] == "shared"
    assert len(ranking) == 3


def test_metrics_use_all_gold_for_ideal_and_ap_denominators():
    ranking = [("positive", 1.0), ("negative", 0.5)]
    result = ranking_metrics(ranking, {"positive", "missing_positive"}, 2)
    assert result["recall@5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ap"] == 0.5
    assert 0.0 < result["ndcg@10"] < 1.0
