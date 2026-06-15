"""FastAPI layer for the triage agent.

One endpoint runs the LangGraph agent and returns a two-audience response:
a patient block (safe, plain language) and a clinical block (signals + the
clinical explanation). The graph is compiled once and reused across requests.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_graph
from src.nodes import patient_message
from src.state import AgentState, UserData

app = FastAPI(title="Triage Agent API")

# Compile once at import; the graph is stateless across requests.
_GRAPH = build_graph()

# Mock CNN fixtures are keyed by these ids.
_KNOWN_VIDEOS = {"video_normal", "video_aom", "video_blurry"}


class TriageRequest(BaseModel):
    video_id: str
    age: int = Field(ge=0, le=120)
    symptoms: list[str] = Field(default_factory=list)
    prior_history: list[str] = Field(default_factory=list)


class PatientView(BaseModel):
    decision: str
    message: str
    patient_explanation: str | None = None
    recapture_reason: str | None = None


class ClinicalView(BaseModel):
    risk_score: float | None = None
    cnn_confidence: float | None = None
    llm_uncertain: bool = False
    similar_case_ids: list[str] = Field(default_factory=list)
    clinical_explanation: str | None = None
    needs_review: bool = False


class TriageResponse(BaseModel):
    patient: PatientView
    clinical: ClinicalView


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    if req.video_id not in _KNOWN_VIDEOS:
        raise HTTPException(status_code=400, detail=f"Unknown video_id: {req.video_id}")

    start = AgentState(
        video_id=req.video_id,
        user_data=UserData(age=req.age, symptoms=req.symptoms, prior_history=req.prior_history),
    )
    final = AgentState(**_GRAPH.invoke(start))

    return TriageResponse(
        patient=PatientView(
            decision=final.decision,
            # Fixed, pre-approved text keyed by decision; never the LLM output, so
            # patient-facing wording stays deterministic and auditable.
            message=patient_message(final.decision, final.recapture_reason),
            patient_explanation=final.patient_explanation,
            recapture_reason=final.recapture_reason,
        ),
        clinical=ClinicalView(
            risk_score=final.risk_score,
            cnn_confidence=final.cnn_confidences.get("diagnosis"),
            llm_uncertain=final.llm_uncertain,
            similar_case_ids=[c.case_id for c in final.similar_cases],
            clinical_explanation=final.clinical_explanation,
            needs_review=final.needs_review,
        ),
    )
