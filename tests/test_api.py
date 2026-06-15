"""Endpoint tests for the triage API. Node 5's LLM is mocked."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import nodes
from src.nodes import LLMOutput


@pytest.fixture
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _mock_llm():
    fake = LLMOutput(
        clinical_explanation="CNN found the finding; case c101 resolved.",
        patient_explanation="A clinician should review this image soon.",
        uncertain=False,
    )
    # _LLM_STRUCTURED is a pydantic RunnableSequence; patch the whole object,
    # not its .invoke method (pydantic blocks attribute patching).
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake
    with patch.object(nodes, "_LLM_STRUCTURED", mock_llm):
        yield


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_triage_normal_case(client):
    r = client.post("/triage", json={"video_id": "video_normal", "age": 30})
    assert r.status_code == 200
    body = r.json()
    assert body["patient"]["decision"]
    assert body["patient"]["message"]
    assert "clinical_explanation" not in body["patient"]
    assert "clinical_explanation" in body["clinical"]
    assert body["clinical"]["needs_review"] is False


def test_triage_blurry_is_recapture(client):
    r = client.post("/triage", json={"video_id": "video_blurry", "age": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["patient"]["decision"] == "re_capture"
    assert body["patient"]["recapture_reason"]


def test_triage_rejects_bad_age(client):
    r = client.post("/triage", json={"video_id": "video_normal", "age": -5})
    assert r.status_code == 422


def test_triage_unknown_video_returns_400(client):
    r = client.post("/triage", json={"video_id": "nope", "age": 30})
    assert r.status_code == 400
