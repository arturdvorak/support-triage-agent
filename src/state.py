"""Shared agent state. Every LangGraph node reads from and writes to this object."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


Visibility = Literal["not_visible", "partially_visible", "fully_visible"]
QualityFlag = Literal["out_of_focus", "too_dark", "earwax", "tubes"]
Diagnosis = Literal["normal", "aom", "chronic_om"]
Decision = Literal[
    "auto_clear",
    "escalate_routine",
    "escalate_urgent",
    "escalate_uncertain",
    "re_capture",
]


class UserData(BaseModel):
    """Patient context passed in with each request."""

    age: int
    symptoms: list[str] = Field(default_factory=list)
    prior_history: list[str] = Field(default_factory=list)


class SimilarCase(BaseModel):
    """One past case returned by the retrieval node."""

    case_id: str
    diagnosis: Diagnosis
    treatment: str
    outcome: str
    similarity: float


class AgentState(BaseModel):
    """State that flows through all 6 nodes. Optional fields start empty and get filled in."""

    # Input
    video_id: str
    user_data: UserData

    # Node 1 - CNN outputs
    visibility: Optional[Visibility] = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    diagnosis: Optional[Diagnosis] = None
    # Confidence per CNN head, e.g. {"visibility": 0.95, "diagnosis": 0.88}
    confidences: dict[str, float] = Field(default_factory=dict)

    # Node 3 - Risk scoring
    risk_score: Optional[float] = None
    risk_confidence: Optional[float] = None

    # Node 4 - Retrieval
    similar_cases: list[SimilarCase] = Field(default_factory=list)

    # Node 5 - LLM
    explanation: Optional[str] = None
    llm_uncertain: bool = False

    # Node 2 or Node 6 - Final routing
    decision: Optional[Decision] = None
    recapture_reason: Optional[str] = None
