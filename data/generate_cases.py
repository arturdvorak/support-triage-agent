"""Generate synthetic past ear-triage cases and write data/cases.json.

Run once by hand: `python -m data.generate_cases`. The output is committed and
loaded into ChromaDB by src/retrieval.py. Uses a fixed random seed so re-running
produces the identical file.

These cases stand in for what would live in PostgreSQL in production. They are
fully synthetic - no real patient data.
"""

import json
import random
from pathlib import Path

# Fixed seed: re-running yields the identical cases.json.
random.seed(42)

N_CASES = 100
OUT_PATH = Path(__file__).parent / "cases.json"

# Age bands and a sample age range for each, so description and age agree.
_AGE_BANDS = {
    "infant": (0, 1),
    "child": (2, 12),
    "teen": (13, 17),
    "adult": (18, 64),
    "senior": (65, 90),
}

# Per-diagnosis templates. Each diagnosis draws its treatment, outcome, and
# symptoms from its own pools so the generated text stays clinically plausible.
_TEMPLATES = {
    "normal": {
        "treatments": ["none"],
        "outcomes": ["resolved", "no follow-up needed"],
        "symptom_pool": [[], [], ["mild_itch"]],
    },
    "aom": {
        "treatments": ["amoxicillin 10d", "amoxicillin 7d", "watchful waiting", "ibuprofen + observation"],
        "outcomes": ["resolved", "resolved after antibiotics"],
        "symptom_pool": [["ear_pain"], ["ear_pain", "fever"], ["ear_pain", "high_fever"], ["high_fever"]],
    },
    "chronic_om": {
        "treatments": ["ENT referral", "tube placement", "antibiotic ear drops", "ongoing monitoring"],
        "outcomes": ["ongoing", "resolved after tubes"],
        "symptom_pool": [["ear_drainage"], ["hearing_loss"], ["ear_pain", "ear_drainage"], ["hearing_loss", "ear_drainage"]],
    },
}

_DX_TEXT = {
    "normal": "Normal eardrum exam",
    "aom": "Acute otitis media (middle ear infection)",
    "chronic_om": "Chronic otitis media",
}


def _describe(diagnosis: str, age: int, symptoms: list[str], treatment: str, outcome: str) -> str:
    """Build the free-text description fed to the embedding model."""
    sx = ", ".join(symptoms) if symptoms else "no notable symptoms"
    tx = "no treatment" if treatment == "none" else treatment
    return (
        f"{_DX_TEXT[diagnosis]} in a {age}-year-old patient presenting with {sx}. "
        f"Managed with {tx}; outcome: {outcome}."
    )


def generate() -> list[dict]:
    diagnoses = list(_TEMPLATES.keys())
    cases: list[dict] = []

    for i in range(N_CASES):
        # Cycle diagnoses so each gets a roughly equal share (34/33/33).
        diagnosis = diagnoses[i % len(diagnoses)]
        tpl = _TEMPLATES[diagnosis]

        age_band = random.choice(list(_AGE_BANDS.keys()))
        low, high = _AGE_BANDS[age_band]
        age = random.randint(low, high)

        symptoms = random.choice(tpl["symptom_pool"])
        treatment = random.choice(tpl["treatments"])
        outcome = random.choice(tpl["outcomes"])

        cases.append({
            "case_id": f"c{i + 1:03d}",
            "diagnosis": diagnosis,
            "description": _describe(diagnosis, age, symptoms, treatment, outcome),
            "treatment": treatment,
            "outcome": outcome,
            "age": age,
            "age_band": age_band,
            "symptoms": symptoms,
        })

    return cases


def main() -> None:
    cases = generate()
    OUT_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Wrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
