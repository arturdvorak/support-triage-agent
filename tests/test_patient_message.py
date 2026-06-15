"""Tests for the fixed patient-message helper."""

import pytest

from src.nodes import patient_message


def test_each_decision_has_a_fixed_message():
    for decision in ["auto_clear", "escalate_routine", "escalate_urgent", "escalate_uncertain"]:
        msg = patient_message(decision, None)
        assert isinstance(msg, str) and msg


def test_recapture_uses_the_reason_text():
    msg = patient_message("re_capture", "image is out of focus")
    assert msg == "image is out of focus"


def test_recapture_with_no_reason_uses_fallback():
    msg = patient_message("re_capture", None)
    assert msg == "Please retake the image."


def test_messages_contain_no_diagnosis_or_drug_terms():
    banned = ["aom", "otitis", "amoxicillin", "infection"]
    for decision in ["auto_clear", "escalate_routine", "escalate_urgent", "escalate_uncertain"]:
        msg = patient_message(decision, None).lower()
        assert not any(term in msg for term in banned)
