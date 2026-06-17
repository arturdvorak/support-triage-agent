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
