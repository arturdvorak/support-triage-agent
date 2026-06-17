"""Retrieval (Node 4) evaluation over the golden set.

Runs the real ChromaDB retrieval for each non-recapture golden case and averages
the ranking metrics. Binary metrics treat grade >= 1 as relevant (the additive
grading lets a same-diagnosis case score 0, so this is meaningful); NDCG uses the
full graded gains.
"""

from statistics import mean

from data.build_golden_dataset import load_golden
from src.retrieval import retrieve_similar
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

K = 3


def run() -> dict:
    cases = [c for c in load_golden() if c.expected_decision != "re_capture"]
    p, r, mrr, ndcg = [], [], [], []
    for c in cases:
        hits = retrieve_similar(c.input.cnn.diagnosis, c.input.age, c.input.symptoms, k=K)
        ids = [h.case_id for h in hits]
        p.append(precision_at_k(ids, c.relevance, K))
        r.append(recall_at_k(ids, c.relevance, K))
        mrr.append(reciprocal_rank(ids, c.relevance))
        ndcg.append(ndcg_at_k(ids, c.relevance, K))
    return {
        "n": len(cases),
        "precision@3": round(mean(p), 3),
        "recall@3": round(mean(r), 3),
        "mrr": round(mean(mrr), 3),
        "ndcg@3": round(mean(ndcg), 3),
    }


def main() -> None:
    metrics = run()
    print("Retrieval (Node 4):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
