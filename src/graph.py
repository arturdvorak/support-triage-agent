"""LangGraph wiring for the 6-node triage workflow."""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.nodes import (
    node1_cnn_inference,
    node2_quality_gate,
    node3_risk_scoring,
    node4_similar_cases,
    node5_llm_explanation,
    node6_escalation,
)


def _route_after_gate(state: AgentState) -> str:
    """If Node 2 set re_capture, end the run early. Otherwise continue to risk scoring."""
    if state.decision == "re_capture":
        return "end"
    return "continue"


def build_graph():
    """Build and compile the triage workflow."""
    g = StateGraph(AgentState)

    g.add_node("node1_cnn", node1_cnn_inference)
    g.add_node("node2_gate", node2_quality_gate)
    g.add_node("node3_risk", node3_risk_scoring)
    g.add_node("node4_cases", node4_similar_cases)
    g.add_node("node5_llm", node5_llm_explanation)
    g.add_node("node6_escalation", node6_escalation)

    g.set_entry_point("node1_cnn")
    g.add_edge("node1_cnn", "node2_gate")

    g.add_conditional_edges(
        "node2_gate",
        _route_after_gate,
        {"end": END, "continue": "node3_risk"},
    )

    g.add_edge("node3_risk", "node4_cases")
    g.add_edge("node4_cases", "node5_llm")
    g.add_edge("node5_llm", "node6_escalation")
    g.add_edge("node6_escalation", END)

    return g.compile()
