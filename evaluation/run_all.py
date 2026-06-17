"""Run the three evaluations and write evaluation/report.md.

The report leads with the headline numbers (end-to-end sensitivity/specificity and
Node 5 hard hallucination rate), then lists the supporting metrics.
"""

from datetime import date
from pathlib import Path

from evaluation import eval_retrieval, eval_end_to_end, eval_node5

_REPORT = Path(__file__).parent / "report.md"


def _render(retrieval: dict, e2e: dict, node5: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Date: {date.today().isoformat()}",
        f"Golden set: {e2e['n']} cases. Small sample, so numbers are directional "
        "with wide error bars.",
        "",
        "## Headline",
        "",
        f"- End-to-end sensitivity: {e2e['sensitivity']} (under-triage {e2e['under_triage']})",
        f"- End-to-end specificity: {e2e['specificity']} (over-triage {e2e['over_triage']})",
        f"- Node 5 hard hallucination rate: {node5['hard_hallucination_rate']} (target 0)",
        "",
        "## End-to-end decision",
        "",
        f"- macro-F1: {e2e['macro_f1']}",
        "- Plain accuracy is intentionally not the lead: under class imbalance it "
        "flatters a majority-class guesser.",
        "",
        "## Retrieval (Node 4)",
        "",
        f"- precision@3: {retrieval['precision@3']}",
        f"- recall@3: {retrieval['recall@3']}",
        f"- mrr: {retrieval['mrr']}",
        f"- ndcg@3: {retrieval['ndcg@3']}",
        "",
        "## Node 5 (LLM explanation)",
        "",
        f"- schema pass rate: {node5['schema_pass_rate']}",
        f"- hard hallucination rate: {node5['hard_hallucination_rate']}",
        f"- soft hallucination rate: {node5.get('soft_hallucination_rate', 'n/a')}",
        f"- judge score (1-5 avg): {node5.get('judge_score_avg', 'n/a')}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(with_judge: bool = True) -> None:
    retrieval = eval_retrieval.run()
    e2e = eval_end_to_end.run()
    node5 = eval_node5.run(with_judge=with_judge)
    report = _render(retrieval, e2e, node5)
    _REPORT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
