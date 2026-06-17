"""The 6 LangGraph nodes for the triage workflow.

Each node is a function: (state) -> dict of fields to merge into state.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.state import AgentState, SimilarCase
from src.mock_services import mock_cnn
from src.retrieval import get_collection, seed_if_empty, retrieve_similar

load_dotenv()

# Safe-default: without an API key, force tracing off so LangChain does not
# try to ship traces and warn on every call.
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"

# Load the synthetic cases into ChromaDB once. Idempotent: skips if already seeded.
seed_if_empty(get_collection())


def node1_cnn_inference(state: AgentState) -> dict:
    """Run the CNN once and store all 3 outputs in state."""
    cnn_output = mock_cnn(state.video_id)
    return {
        "visibility": cnn_output["visibility"],
        "quality_flags": cnn_output["quality_flags"],
        "diagnosis": cnn_output["diagnosis"],
        "cnn_confidences": cnn_output["confidences"],
    }


# User-facing strings for each flag. Kept separate so the UI copy is easy to change.
_FLAG_REASONS = {
    "out_of_focus": "image is out of focus",
    "too_dark": "image is too dark",
    "earwax": "too much earwax blocking the view",
    "tubes": "medical tubes blocking the view",
}


def node2_quality_gate(state: AgentState) -> dict:
    """Force re_capture if the eardrum is not fully visible or any quality flag fired."""
    reasons: list[str] = []

    if state.visibility != "fully_visible":
        reasons.append("eardrum is not fully visible")

    for flag in state.quality_flags:
        reasons.append(_FLAG_REASONS.get(flag, flag))

    if not reasons:
        return {}

    return {
        "decision": "re_capture",
        "recapture_reason": "; ".join(reasons),
    }


# Base clinical risk per diagnosis. Tunable - intentionally pessimistic on chronic.
_BASE_RISK = {"normal": 0.10, "aom": 0.60, "chronic_om": 0.70}


def node3_risk_scoring(state: AgentState) -> dict:
    """Combine diagnosis + patient context into a 0-1 risk score."""
    risk = _BASE_RISK[state.diagnosis]

    age = state.user_data.age
    if age < 2:
        risk += 0.20
    elif age > 65:
        risk += 0.10

    symptoms = set(state.user_data.symptoms)
    if symptoms & {"fever", "high_fever"}:
        risk += 0.20
    if "severe_pain" in symptoms:
        risk += 0.15

    if state.user_data.prior_history:
        risk += 0.10

    risk = max(0.0, min(1.0, risk))

    return {
        "risk_score": risk,
        "risk_confidence": state.cnn_confidences.get("diagnosis", 0.0),
    }


def node4_similar_cases(state: AgentState) -> dict:
    """Look up past cases with the same diagnosis to ground the LLM explanation."""
    return {
        "similar_cases": retrieve_similar(
            diagnosis=state.diagnosis,
            age=state.user_data.age,
            symptoms=state.user_data.symptoms,
        )
    }


class LLMOutput(BaseModel):
    """Structured response we ask Claude to return."""

    clinical_explanation: str = Field(
        description="Clinical explanation for a doctor. 2-3 sentences. May name the diagnosis and reference past case IDs."
    )
    patient_explanation: str = Field(
        description="Plain-language explanation for the patient. No diagnosis name, no drug name, no case IDs, no numbers. Always points to a clinician."
    )
    uncertain: bool = Field(
        description="True if the signals disagree or the model is not confident."
    )


# Built once at import time. temperature=0 for reproducible explanations.
_LLM = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
_LLM_STRUCTURED = _LLM.with_structured_output(LLMOutput)


def _format_cases(cases: list[SimilarCase]) -> str:
    return "\n".join(
        f"- {c.case_id} | dx={c.diagnosis} | tx={c.treatment} | outcome={c.outcome} | sim={c.similarity:.2f}"
        for c in cases
    )


def node5_llm_explanation(state: AgentState) -> dict:
    """Ask Claude for a clinical explanation and a separate patient-safe explanation."""
    prompt = f"""You are a clinical assistant explaining an ear triage decision.

CNN diagnosis: {state.diagnosis} (confidence {state.cnn_confidences.get('diagnosis', 0):.2f})
Risk score: {state.risk_score:.2f}
Patient: age {state.user_data.age}, symptoms {state.user_data.symptoms or 'none'}, prior history {state.user_data.prior_history or 'none'}

Similar past cases:
{_format_cases(state.similar_cases)}

Write two explanations:
1. clinical_explanation (for a clinician): 2-3 sentences. Reference at least one case by its ID. Do not invent findings.
2. patient_explanation (for the patient): 1-2 sentences in plain language. Do NOT name the diagnosis, do NOT name any drug, do NOT mention case IDs or numbers. Always suggest seeing a clinician.

Set uncertain=true if the signals disagree or the diagnosis confidence looks low."""

    result = _LLM_STRUCTURED.invoke(prompt)
    return {
        "clinical_explanation": result.clinical_explanation,
        "patient_explanation": result.patient_explanation,
        "llm_uncertain": result.uncertain,
    }


# Thresholds from the Q2 decision table. Centralized so they are easy to tune.
_HIGH_CONF = 0.90
_LOW_CONF = 0.70
_HIGH_RISK = 0.80


# Fixed, pre-approved patient-facing text per decision. Deterministic by design:
# a regulator must know the exact words a patient can ever see. Never LLM-generated.
_PATIENT_MESSAGE = {
    "auto_clear": "No signs that need follow-up were found in this image. If symptoms continue, see a clinician.",
    "escalate_routine": "We recommend booking a non-urgent visit with a clinician to review this.",
    "escalate_urgent": "Please seek care promptly. Contact a clinician today.",
    "escalate_uncertain": "We could not reach a clear result. A clinician should review this.",
}


def patient_message(decision: str, recapture_reason: str | None) -> str:
    """Return the fixed patient-facing message for a decision.

    re_capture has no entry in the map; it reuses the Node 2 recapture_reason text.
    """
    if decision == "re_capture":
        return recapture_reason or "Please retake the image."
    return _PATIENT_MESSAGE[decision]


def node6_escalation(state: AgentState) -> dict:
    """Rule-based final decision over the 3 signals: CNN, retrieved cases, LLM."""
    cnn_conf = state.cnn_confidences.get("diagnosis", 0.0)

    # Cases vote: majority must match the CNN diagnosis to count as agreement.
    matching = sum(1 for c in state.similar_cases if c.diagnosis == state.diagnosis)
    cases_agree = matching > len(state.similar_cases) / 2

    # Any uncertainty signal -> defer to a human. Bias toward escalation.
    # llm_uncertain is intentionally excluded: routing stays deterministic and
    # auditable. The flag remains a Node 5 output, it just does not route.
    if cnn_conf < _LOW_CONF or not cases_agree:
        return {"decision": "escalate_uncertain"}

    if state.diagnosis != "normal":
        symptoms = set(state.user_data.symptoms)
        severe = bool(symptoms & {"high_fever", "severe_pain"}) or state.risk_score >= _HIGH_RISK
        return {"decision": "escalate_urgent" if severe else "escalate_routine"}

    if cnn_conf > _HIGH_CONF:
        return {"decision": "auto_clear"}

    # Normal diagnosis but confidence in the middle band - safer to escalate.
    return {"decision": "escalate_uncertain"}
