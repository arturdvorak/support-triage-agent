"""Node 6 must route on deterministic signals only, ignoring llm_uncertain."""

from src.state import AgentState, UserData, SimilarCase
from src.nodes import node6_escalation


def _normal_state(llm_uncertain: bool) -> AgentState:
    return AgentState(
        video_id="t",
        user_data=UserData(age=30, symptoms=[]),
        diagnosis="normal",
        cnn_confidences={"diagnosis": 0.95},
        similar_cases=[
            SimilarCase(case_id="c1", diagnosis="normal", treatment="none",
                        outcome="resolved", similarity=0.9)
        ],
        llm_uncertain=llm_uncertain,
    )


def test_node6_ignores_llm_uncertain():
    # High CNN confidence, cases agree, normal diagnosis -> auto_clear,
    # even though the LLM flagged uncertainty.
    assert node6_escalation(_normal_state(True))["decision"] == "auto_clear"
    assert node6_escalation(_normal_state(False))["decision"] == "auto_clear"
