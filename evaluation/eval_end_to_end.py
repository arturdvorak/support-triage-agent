"""End-to-end decision evaluation.

Runs the deterministic node path (Node 2 gate, Node 3 risk, Node 4 retrieval,
Node 6 decision) on each golden case and compares the decision to the expected
one. Node 5 is skipped: llm_uncertain no longer routes, so no LLM call is needed
and the result is reproducible.

Known limitation: Node 4 filters retrieval by diagnosis, so cases_agree in Node 6
is always true and the not-cases_agree term never fires. escalate_uncertain is
driven by cnn_conf < 0.70 plus the normal mid-confidence band.
"""

from data.build_golden_dataset import load_golden, GoldenCase
from src.state import AgentState, UserData
from src.nodes import (
    node2_quality_gate,
    node3_risk_scoring,
    node4_similar_cases,
    node6_escalation,
)
from evaluation.metrics import macro_f1, sensitivity_specificity


def _build_state(case: GoldenCase) -> AgentState:
    cnn = case.input.cnn
    return AgentState(
        video_id=case.id,
        user_data=UserData(
            age=case.input.age,
            symptoms=case.input.symptoms,
            prior_history=case.input.prior_history,
        ),
        visibility=cnn.visibility,
        quality_flags=cnn.quality_flags,
        diagnosis=cnn.diagnosis,
        cnn_confidences={"diagnosis": cnn.diagnosis_confidence},
    )


def predict_decision(case: GoldenCase) -> str:
    state = _build_state(case)
    gate = node2_quality_gate(state)
    if gate.get("decision"):
        return gate["decision"]
    state = state.model_copy(update=node3_risk_scoring(state))
    state = state.model_copy(update=node4_similar_cases(state))
    return node6_escalation(state)["decision"]


def run() -> dict:
    cases = load_golden()
    expected = [c.expected_decision for c in cases]
    predicted = [predict_decision(c) for c in cases]
    ss = sensitivity_specificity(expected, predicted)
    return {
        "n": len(cases),
        "sensitivity": round(ss["sensitivity"], 3),
        "specificity": round(ss["specificity"], 3),
        "under_triage": round(ss["under_triage"], 3),
        "over_triage": round(ss["over_triage"], 3),
        "macro_f1": round(macro_f1(expected, predicted), 3),
    }


def main() -> None:
    metrics = run()
    print("End-to-end decision:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
