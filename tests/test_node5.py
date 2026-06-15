"""Node 5 returns both a clinical and a patient explanation."""

from unittest.mock import MagicMock, patch

from src import nodes
from src.nodes import node5_llm_explanation, LLMOutput
from src.state import AgentState, UserData, SimilarCase


def _state():
    return AgentState(
        video_id="video_aom",
        user_data=UserData(age=1, symptoms=["high_fever"], prior_history=[]),
        diagnosis="aom",
        cnn_confidences={"diagnosis": 0.88},
        risk_score=0.88,
        similar_cases=[SimilarCase(case_id="c101", diagnosis="aom", treatment="amoxicillin 10d", outcome="resolved", similarity=0.9)],
    )


def test_node5_returns_both_explanations():
    fake = LLMOutput(
        clinical_explanation="CNN found aom; case c101 resolved with amoxicillin.",
        patient_explanation="The image showed something that a clinician should review soon.",
        uncertain=False,
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake
    with patch.object(nodes, "_LLM_STRUCTURED", mock_llm):
        out = node5_llm_explanation(_state())
    assert out["clinical_explanation"]
    assert out["patient_explanation"]
    assert out["llm_uncertain"] is False
