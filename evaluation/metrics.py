"""Pure metric helpers for the evaluation scripts. Stdlib only, no heavy imports.

Retrieval metrics take a ranked list of retrieved case ids and a relevance map
(case_id -> graded relevance). `rel` is the minimum grade that counts as relevant
for the binary metrics; NDCG uses the full graded gains.
"""

import math


def precision_at_k(retrieved: list[str], relevance: dict[str, int], k: int = 3, rel: int = 1) -> float:
    top = retrieved[:k]
    hits = sum(1 for cid in top if relevance.get(cid, 0) >= rel)
    return hits / k


def recall_at_k(retrieved: list[str], relevance: dict[str, int], k: int = 3, rel: int = 1) -> float:
    total = sum(1 for g in relevance.values() if g >= rel)
    if total == 0:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if relevance.get(cid, 0) >= rel)
    return hits / total


def reciprocal_rank(retrieved: list[str], relevance: dict[str, int], rel: int = 1) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if relevance.get(cid, 0) >= rel:
            return 1.0 / i
    return 0.0


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int = 3) -> float:
    grades = [relevance.get(cid, 0) for cid in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(grades) / idcg


_ESCALATE = {"escalate_routine", "escalate_urgent", "escalate_uncertain"}
_CLEAR = {"auto_clear"}


def _per_class_f1(expected: list[str], predicted: list[str], cls: str) -> float:
    tp = sum(1 for e, p in zip(expected, predicted) if e == cls and p == cls)
    fp = sum(1 for e, p in zip(expected, predicted) if e != cls and p == cls)
    fn = sum(1 for e, p in zip(expected, predicted) if e == cls and p != cls)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(expected: list[str], predicted: list[str], labels: list[str] | None = None) -> float:
    """Average per-class F1, each class weighted equally. Defaults to the classes
    present in `expected` so unseen labels do not drag the average to zero."""
    labels = labels if labels is not None else sorted(set(expected))
    if not labels:
        return 0.0
    return sum(_per_class_f1(expected, predicted, c) for c in labels) / len(labels)


def sensitivity_specificity(expected: list[str], predicted: list[str]) -> dict[str, float]:
    """Collapse decisions to escalate-vs-clear (re_capture excluded) and return
    sensitivity, specificity, and their error versions (under/over-triage)."""
    pos = [(e, p) for e, p in zip(expected, predicted) if e in _ESCALATE]
    neg = [(e, p) for e, p in zip(expected, predicted) if e in _CLEAR]

    sensitivity = (sum(1 for _, p in pos if p in _ESCALATE) / len(pos)) if pos else 0.0
    specificity = (sum(1 for _, p in neg if p in _CLEAR) / len(neg)) if neg else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "under_triage": 1.0 - sensitivity,
        "over_triage": 1.0 - specificity,
    }
