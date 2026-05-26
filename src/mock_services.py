"""Stand-in functions for the real CNN model and ChromaDB retrieval.

Keep the I/O shape close to what the real services would return so swapping
them later is just changing the function body, not the call site.
"""

from src.state import SimilarCase


_CNN_FIXTURES: dict[str, dict] = {
    "video_normal": {
        "visibility": "fully_visible",
        "quality_flags": [],
        "diagnosis": "normal",
        "confidences": {"visibility": 0.97, "diagnosis": 0.93},
    },
    "video_aom": {
        "visibility": "fully_visible",
        "quality_flags": [],
        "diagnosis": "aom",
        "confidences": {"visibility": 0.95, "diagnosis": 0.88},
    },
    "video_blurry": {
        # Quality flag fires -> Node 2 should route directly to re_capture.
        "visibility": "fully_visible",
        "quality_flags": ["out_of_focus"],
        "diagnosis": "normal",
        "confidences": {"visibility": 0.60, "diagnosis": 0.40},
    },
}


def mock_cnn(video_id: str) -> dict:
    """Return canned CNN output for a known video_id."""
    if video_id not in _CNN_FIXTURES:
        raise ValueError(f"Unknown mock video_id: {video_id}")
    return _CNN_FIXTURES[video_id]


_RETRIEVAL_FIXTURES: dict[str, list[SimilarCase]] = {
    "normal": [
        SimilarCase(case_id="c001", diagnosis="normal", treatment="none", outcome="resolved", similarity=0.92),
        SimilarCase(case_id="c002", diagnosis="normal", treatment="none", outcome="resolved", similarity=0.88),
        SimilarCase(case_id="c003", diagnosis="normal", treatment="none", outcome="resolved", similarity=0.85),
    ],
    "aom": [
        SimilarCase(case_id="c101", diagnosis="aom", treatment="amoxicillin 10d", outcome="resolved", similarity=0.91),
        SimilarCase(case_id="c102", diagnosis="aom", treatment="amoxicillin 7d", outcome="resolved", similarity=0.87),
        SimilarCase(case_id="c103", diagnosis="aom", treatment="watchful waiting", outcome="resolved", similarity=0.83),
    ],
    "chronic_om": [
        SimilarCase(case_id="c201", diagnosis="chronic_om", treatment="ENT referral", outcome="ongoing", similarity=0.90),
        SimilarCase(case_id="c202", diagnosis="chronic_om", treatment="tube placement", outcome="resolved", similarity=0.86),
        SimilarCase(case_id="c203", diagnosis="chronic_om", treatment="ENT referral", outcome="ongoing", similarity=0.82),
    ],
}


def mock_retrieval(diagnosis: str) -> list[SimilarCase]:
    """Return 3 canned similar cases matching the given diagnosis."""
    return _RETRIEVAL_FIXTURES.get(diagnosis, [])
