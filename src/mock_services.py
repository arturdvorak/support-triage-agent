"""Stand-in for the real CNN model.

Keeps the I/O shape close to what the real service would return so swapping it
later is just changing the function body, not the call site. Case retrieval has
moved to src/retrieval.py (real ChromaDB), so only the CNN mock lives here now.
"""

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
