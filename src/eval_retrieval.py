"""Score retrieval quality with precision@k and NDCG@k.

Ground truth is rule-based and graded, computed from data/cases.json (no
hand-labeling). For a query, each case is graded:
  2 - same diagnosis AND same age band AND symptom overlap (strong match)
  1 - same diagnosis AND (age band OR symptom overlap)        (partial match)
  0 - otherwise                                               (not relevant)
Precision@k treats grade >= 1 as relevant. NDCG@k uses the full 2/1/0 grades,
so it rewards ranking strong matches above partial ones. Run with:
`python -m src.eval_retrieval`.
"""

import json
import math
from pathlib import Path

from src.retrieval import retrieve_similar

_CASES_PATH = Path(__file__).parent.parent / "data" / "cases.json"
K = 3

# Same ranges as data/generate_cases.py, so a query age maps to the same band.
_AGE_BANDS = {
    "infant": (0, 1),
    "child": (2, 12),
    "teen": (13, 17),
    "adult": (18, 64),
    "senior": (65, 90),
}

# Each query is (diagnosis, age, symptoms), spanning diagnoses, age bands, and
# symptom patterns including the empty-symptom case.
_TEST_QUERIES = [
    ("normal", 30, []),
    ("normal", 1, []),
    ("normal", 70, ["mild_itch"]),
    ("aom", 1, ["ear_pain", "high_fever"]),
    ("aom", 4, ["ear_pain"]),
    ("aom", 30, ["ear_pain", "fever"]),
    ("aom", 70, ["high_fever"]),
    ("chronic_om", 1, ["ear_drainage"]),
    ("chronic_om", 8, ["ear_pain", "ear_drainage"]),
    ("chronic_om", 70, ["hearing_loss"]),
    ("chronic_om", 40, ["hearing_loss", "ear_drainage"]),
    ("normal", 15, []),
    ("aom", 10, ["ear_pain", "high_fever"]),
    ("chronic_om", 30, ["ear_drainage"]),
]


def age_band(age: int) -> str:
    for band, (low, high) in _AGE_BANDS.items():
        if low <= age <= high:
            return band
    return "unknown"


def grade(dx: str, band: str, symptoms: list[str], case: dict) -> int:
    """Relevance of one case to a query: 2 (strong), 1 (partial), 0 (none).

    Diagnosis must match. Then band match and symptom match each add a point,
    so both -> 2, one -> 1. Two symptom-free presentations count as a match.
    """
    if case["diagnosis"] != dx:
        return 0
    band_match = case["age_band"] == band
    symptom_match = bool(set(symptoms) & set(case["symptoms"])) or (
        not symptoms and not case["symptoms"]
    )
    return int(band_match) + int(symptom_match)


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain: each grade divided by log2(position + 1)."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg(returned: list[int], ideal: list[int]) -> float:
    """DCG of returned grades over DCG of the best possible ordering, in [0, 1]."""
    best = dcg(ideal)
    return dcg(returned) / best if best else 0.0


def main() -> None:
    cases = json.loads(_CASES_PATH.read_text())
    sum_p = sum_ndcg = 0.0

    print(f"\nRetrieval eval over {len(_TEST_QUERIES)} queries (k={K}):\n")
    header = f"{'diagnosis':<11} {'age':>3} {'symptoms':<24} {'grades':<8} {'P@k':>6} {'NDCG@k':>7}"
    print(header)
    print("-" * len(header))

    for dx, age, symptoms in _TEST_QUERIES:
        band = age_band(age)
        all_grades = [grade(dx, band, symptoms, c) for c in cases]
        by_id = {c["case_id"]: g for c, g in zip(cases, all_grades)}

        hits = retrieve_similar(dx, age, symptoms, k=K)
        returned = [by_id[h.case_id] for h in hits]

        p = sum(g >= 1 for g in returned) / len(returned)
        n = ndcg(returned, sorted(all_grades, reverse=True)[:K])
        sum_p += p
        sum_ndcg += n

        sx_str = ", ".join(symptoms) if symptoms else "none"
        grades_str = ",".join(str(g) for g in returned)
        print(f"{dx:<11} {age:>3} {sx_str:<24} {grades_str:<8} {p:>6.2f} {n:>7.2f}")

    q = len(_TEST_QUERIES)
    print("-" * len(header))
    print(f"\nMean precision@{K}: {sum_p / q:.3f}")
    print(f"Mean NDCG@{K}     : {sum_ndcg / q:.3f}")


if __name__ == "__main__":
    main()
