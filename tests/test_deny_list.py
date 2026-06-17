"""The deny-list scanner flags banned content in patient-facing text."""

from evaluation.deny_list import deny_list_hits


def test_clean_text_passes():
    text = "A clinician should look at this image soon to be safe."
    assert deny_list_hits(text) == []


def test_diagnosis_word_is_flagged():
    assert "otitis" in deny_list_hits("This looks like otitis media.")


def test_drug_name_is_flagged():
    assert "amoxicillin" in deny_list_hits("They may prescribe amoxicillin.")


def test_case_id_is_flagged():
    assert "case_id:c014" in deny_list_hits("Similar to case c014 in our records.")


def test_number_is_flagged():
    assert "number:88" in deny_list_hits("Confidence was about 88 percent.")
