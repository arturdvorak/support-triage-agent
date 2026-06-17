"""The end-to-end path is deterministic and produces expected decisions on
two known sample rows. (Row ids assume the committed golden.json.)"""

from data.build_golden_dataset import load_golden
from evaluation.eval_end_to_end import predict_decision


def _by_id():
    return {c.id: c for c in load_golden()}


def test_end_to_end_is_deterministic():
    cases = _by_id()
    for cid in ("g034", "g040"):
        case = cases[cid]
        assert predict_decision(case) == predict_decision(case)


def test_known_sample_decisions():
    cases = _by_id()
    # g040 = not_visible image -> Node 2 forces re_capture.
    assert predict_decision(cases["g040"]) == "re_capture"
    # g034 = infant + high fever + aom (conf 0.88) -> urgent escalation.
    assert predict_decision(cases["g034"]) == "escalate_urgent"
