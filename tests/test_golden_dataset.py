"""The committed golden.json is well-formed and references real corpus cases."""

import json
from pathlib import Path

from data.build_golden_dataset import GoldenCase

_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN = _ROOT / "data" / "golden.json"
_CASES = _ROOT / "data" / "cases.json"


def _rows():
    return json.loads(_GOLDEN.read_text())


def test_golden_loads_and_validates():
    rows = _rows()
    assert len(rows) == 40
    for r in rows:
        GoldenCase(**r)  # raises on any invalid row


def test_relevance_ids_exist_in_corpus():
    corpus_ids = {c["case_id"] for c in json.loads(_CASES.read_text())}
    for r in _rows():
        for cid in r["relevance"]:
            assert cid in corpus_ids


def test_has_clear_escalate_and_recapture_cases():
    decisions = [r["expected_decision"] for r in _rows()]
    assert "auto_clear" in decisions
    assert any(d.startswith("escalate") for d in decisions)
    assert "re_capture" in decisions
