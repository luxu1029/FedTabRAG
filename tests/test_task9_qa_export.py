from src.evaluation.finalize_structure_branch_and_plan_qa import classify, recommended_track


def retrieval(gold_hit=True, evidence_id="gold", text="values 10 and 20"):
    return {
        "gold_hit": gold_hit,
        "retrieved": [{"evidence_id": evidence_id, "text": text}],
    }


def qa(numerical_em=False, parsed_prediction={"value": 1.0}):
    return {"numerical_em": numerical_em, "parsed_prediction": parsed_prediction}


def test_classification_preserves_success_and_retrieval_boundaries():
    positive = {"row": ["gold"], "cell": []}
    assert classify(retrieval(), qa(numerical_em=True), positive, "add(10,20)") == "success"
    assert classify(retrieval(gold_hit=False), qa(), positive, "add(10,20)") == "retrieval_failure"


def test_gold_hit_wrong_taxonomy():
    assert classify(retrieval(evidence_id="table"), qa(), {"row": ["gold"]},
                    "add(10, 20)") == "structure_localization_failure"
    assert classify(retrieval(), qa(parsed_prediction=None), {"row": ["gold"]},
                    "add(10, 20)") == "format_matching_failure"
    assert classify(retrieval(text="only 10"), qa(), {"row": ["gold"]},
                    "add(10, 20)") == "number_extraction_failure"
    assert classify(retrieval(), qa(), {"row": ["gold"]},
                    "add(10, 20)") == "operation_failure"


def test_recommended_tracks_do_not_claim_program_results():
    assert recommended_track("operation_failure") == "program_generation_or_numeric_reasoning"
    assert recommended_track("format_matching_failure") == "answer_format_and_numeric_parser"
