"""Deny-list scanner for patient-facing text.

A hard hallucination is any banned item appearing in the patient explanation:
a diagnosis word, a drug/treatment word, a case-id reference, or any digit.
Plain code so it is auditable. Used for measurement here; the same idea will
back the runtime guardrail in a later spec.
"""

import re

_DIAGNOSIS_WORDS = ["aom", "otitis", "chronic_om", "infection"]
_DRUG_WORDS = ["amoxicillin", "ibuprofen", "antibiotic", "drops", "tube"]
_CASE_ID = re.compile(r"\bc\d{1,3}\b")
_NUMBER = re.compile(r"\d+")


def deny_list_hits(text: str) -> list[str]:
    """Return a list of banned items found in `text` (empty if clean)."""
    hits: list[str] = []
    low = text.lower()

    for word in _DIAGNOSIS_WORDS + _DRUG_WORDS:
        if word in low:
            hits.append(word)

    for m in _CASE_ID.findall(low):
        hits.append(f"case_id:{m}")

    # Strip case ids first so their digits are not double-counted as numbers.
    low_no_ids = _CASE_ID.sub(" ", low)
    for m in _NUMBER.findall(low_no_ids):
        hits.append(f"number:{m}")

    return hits
