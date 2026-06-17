"""Deterministic Node 5 scoring (schema pass + hard hallucination) over a fixture
cache. No LLM and no network needed for this test."""

import pytest

from evaluation.eval_node5 import score_deterministic


def test_score_deterministic_counts_schema_and_hard():
    outputs = {
        # valid + clean patient text
        "g1": {"clinical_explanation": "CNN finding; case c014 resolved.",
               "patient_explanation": "Please see a clinician soon to be safe.",
               "uncertain": False},
        # valid but patient text leaks a drug name and a number -> hard hallucination
        "g2": {"clinical_explanation": "aom likely.",
               "patient_explanation": "They gave amoxicillin for 7 days.",
               "uncertain": False},
        # generation failed -> not schema-valid
        "g3": {"error": "timeout"},
    }
    m = score_deterministic(outputs)
    assert m["n_generated"] == 3
    assert m["schema_pass_rate"] == pytest.approx(2 / 3)   # g1, g2 valid; g3 not
    assert m["hard_hallucination_rate"] == pytest.approx(0.5)  # of 2 valid, g2 hits
