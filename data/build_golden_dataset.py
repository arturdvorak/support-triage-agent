"""Build data/golden.json: the evaluation golden set (40 cases).

Each case has injected CNN output plus patient context, an expected decision from
a reference policy, and graded relevance labels. The reference policy is authored
from clinical reasoning and is intentionally independent of Node 6's code, so the
end-to-end eval can catch rule bugs instead of grading the code against itself.

Run once by hand: `python -m data.build_golden_dataset`. Fixed seed -> identical
output. If data/cases.json grows, re-run so relevance grades refresh.
"""

import json
import random
from pathlib import Path

from pydantic import BaseModel

from src.state import Visibility, QualityFlag, Diagnosis, Decision

random.seed(7)

_CASES_PATH = Path(__file__).parent / "cases.json"
_OUT_PATH = Path(__file__).parent / "golden.json"

_AGE_BANDS = {"infant": (0, 1), "child": (2, 12), "teen": (13, 17), "adult": (18, 64), "senior": (65, 90)}
_SEVERE = {"high_fever", "severe_pain"}
_SX_POOL = {
    "normal": [[], [], ["mild_itch"]],
    "aom": [["ear_pain"], ["ear_pain", "fever"], ["ear_pain", "high_fever"], ["high_fever"]],
    "chronic_om": [["ear_drainage"], ["hearing_loss"], ["ear_pain", "ear_drainage"], ["hearing_loss", "ear_drainage"]],
}


class GoldenCNN(BaseModel):
    visibility: Visibility
    quality_flags: list[QualityFlag] = []
    diagnosis: Diagnosis
    diagnosis_confidence: float


class GoldenInput(BaseModel):
    age: int
    symptoms: list[str] = []
    prior_history: list[str] = []
    cnn: GoldenCNN


class GoldenCase(BaseModel):
    id: str
    input: GoldenInput
    expected_decision: Decision
    relevance: dict[str, int] = {}


def reference_policy(cnn: dict, age: int, symptoms: list[str]) -> str:
    """Expected decision from clinical reasoning, independent of Node 6's code."""
    if cnn["visibility"] != "fully_visible" or cnn["quality_flags"]:
        return "re_capture"
    conf = cnn["diagnosis_confidence"]
    dx = cnn["diagnosis"]
    if conf < 0.70:
        return "escalate_uncertain"
    if dx == "normal":
        return "auto_clear" if conf >= 0.90 else "escalate_uncertain"
    if age < 2 or (set(symptoms) & _SEVERE):
        return "escalate_urgent"
    return "escalate_routine"


def grade_relevance(corpus: list[dict], dx: str, age: int, symptoms: list[str]) -> dict[str, int]:
    """Grade same-diagnosis corpus cases with the additive rule from
    src/eval_retrieval.py: 2 = age band AND symptom, 1 = exactly one of the two,
    0 = neither or a different diagnosis (not stored). Reuses the existing grader
    so both evals agree on what 'relevant' means.

    Local import: src.eval_retrieval pulls in ChromaDB at module load, so we defer
    it to generation time and keep this module's import light for fast tests.
    """
    from src.eval_retrieval import grade, age_band

    band = age_band(age)
    out: dict[str, int] = {}
    for c in corpus:
        g = grade(dx, band, symptoms, c)
        if g >= 1:
            out[c["case_id"]] = g
    return out


def _near_cases() -> list[tuple[int, list[str], dict]]:
    diagnoses = ["normal", "aom", "chronic_om"]
    rows = []
    for i in range(30):
        dx = diagnoses[i % 3]
        band = random.choice(list(_AGE_BANDS))
        lo, hi = _AGE_BANDS[band]
        age = random.randint(lo, hi)
        symptoms = random.choice(_SX_POOL[dx])
        conf = round(random.uniform(0.85, 0.97), 2)
        cnn = {"visibility": "fully_visible", "quality_flags": [], "diagnosis": dx, "diagnosis_confidence": conf}
        rows.append((age, symptoms, cnn))
    return rows


def _edge_cases() -> list[tuple[int, list[str], dict]]:
    return [
        (5, ["ear_pain"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "aom", "diagnosis_confidence": 0.55}),
        (40, [], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "normal", "diagnosis_confidence": 0.62}),
        (70, ["hearing_loss"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "chronic_om", "diagnosis_confidence": 0.60}),
        (1, ["high_fever", "ear_pain"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "aom", "diagnosis_confidence": 0.88}),
        (0, ["high_fever"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "aom", "diagnosis_confidence": 0.90}),
        (72, ["ear_drainage"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "chronic_om", "diagnosis_confidence": 0.91}),
        (68, ["hearing_loss"], {"visibility": "fully_visible", "quality_flags": [], "diagnosis": "chronic_om", "diagnosis_confidence": 0.93}),
        (30, [], {"visibility": "partially_visible", "quality_flags": [], "diagnosis": "normal", "diagnosis_confidence": 0.80}),
        (25, ["ear_pain"], {"visibility": "fully_visible", "quality_flags": ["out_of_focus"], "diagnosis": "aom", "diagnosis_confidence": 0.50}),
        (3, [], {"visibility": "not_visible", "quality_flags": ["too_dark"], "diagnosis": "normal", "diagnosis_confidence": 0.40}),
    ]


def build() -> list[GoldenCase]:
    corpus = json.loads(_CASES_PATH.read_text())
    rows = _near_cases() + _edge_cases()
    cases: list[GoldenCase] = []
    for i, (age, symptoms, cnn) in enumerate(rows, start=1):
        decision = reference_policy(cnn, age, symptoms)
        relevance = {} if decision == "re_capture" else grade_relevance(corpus, cnn["diagnosis"], age, symptoms)
        cases.append(GoldenCase(
            id=f"g{i:03d}",
            input=GoldenInput(age=age, symptoms=symptoms, prior_history=[], cnn=GoldenCNN(**cnn)),
            expected_decision=decision,
            relevance=relevance,
        ))
    return cases


def main() -> None:
    cases = build()
    _OUT_PATH.write_text(json.dumps([c.model_dump() for c in cases], indent=2))
    print(f"Wrote {len(cases)} golden cases to {_OUT_PATH}")


if __name__ == "__main__":
    main()
