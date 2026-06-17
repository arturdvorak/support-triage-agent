"""Node 5 (LLM explanation) evaluation.

Generates explanations once (cached to node5_outputs.json), then scores them.
Deterministic metrics (schema pass rate, hard hallucination rate) need no LLM and
are unit-tested. The judge metrics (1-5 faithfulness score, soft hallucination
rate) use a separate Claude call per case.
"""

import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

from data.build_golden_dataset import load_golden
from src.nodes import node3_risk_scoring, node4_similar_cases, node5_llm_explanation
from evaluation.eval_end_to_end import _build_state
from evaluation.deny_list import deny_list_hits

_CACHE_PATH = Path(__file__).parent / "node5_outputs.json"


def _valid(out: dict) -> bool:
    return (
        isinstance(out, dict)
        and isinstance(out.get("clinical_explanation"), str) and out["clinical_explanation"].strip() != ""
        and isinstance(out.get("patient_explanation"), str) and out["patient_explanation"].strip() != ""
        and isinstance(out.get("uncertain"), bool)
    )


def score_deterministic(outputs: dict) -> dict:
    total = len(outputs)
    valid = [o for o in outputs.values() if _valid(o)]
    schema_pass_rate = len(valid) / total if total else 0.0
    hard = sum(1 for o in valid if deny_list_hits(o["patient_explanation"]))
    hard_rate = hard / len(valid) if valid else 0.0
    return {
        "n_generated": total,
        # 6 decimals (not 3): the unit test compares against pytest.approx(2/3),
        # whose default tolerance is tighter than a 3-decimal round would satisfy.
        "schema_pass_rate": round(schema_pass_rate, 6),
        "hard_hallucination_rate": round(hard_rate, 6),
    }


def _prepared_state(case):
    state = _build_state(case)
    state = state.model_copy(update=node3_risk_scoring(state))
    state = state.model_copy(update=node4_similar_cases(state))
    return state


def generate_outputs(cases) -> dict:
    outputs: dict = {}
    for c in cases:
        if c.expected_decision == "re_capture":
            continue  # re_capture exits before Node 5
        try:
            res = node5_llm_explanation(_prepared_state(c))
            outputs[c.id] = {
                "clinical_explanation": res["clinical_explanation"],
                "patient_explanation": res["patient_explanation"],
                "uncertain": res["llm_uncertain"],
            }
        except Exception as exc:  # record failures so schema pass rate can see them
            outputs[c.id] = {"error": str(exc)}
    return outputs


def load_or_generate(cases, use_cache: bool = True) -> dict:
    if use_cache and _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    outputs = generate_outputs(cases)
    _CACHE_PATH.write_text(json.dumps(outputs, indent=2))
    return outputs


class JudgeOutput(BaseModel):
    score: int = Field(description="1-5: faithfulness to the given inputs and clarity")
    soft_hallucination: bool = Field(description="True if any claim is not supported by the inputs")


_JUDGE = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(JudgeOutput)


def _context(case) -> str:
    state = _prepared_state(case)
    cases_txt = "\n".join(
        f"- {c.case_id} dx={c.diagnosis} tx={c.treatment} outcome={c.outcome}"
        for c in state.similar_cases
    )
    return (
        f"Diagnosis: {state.diagnosis} (confidence {case.input.cnn.diagnosis_confidence})\n"
        f"Risk score: {state.risk_score}\n"
        f"Similar past cases:\n{cases_txt}"
    )


def judge_outputs(cases, outputs: dict) -> dict:
    by_id = {c.id: c for c in cases}
    scores, soft = [], []
    for cid, out in outputs.items():
        if not _valid(out):
            continue
        prompt = (
            "You are grading a clinical explanation written for a doctor.\n\n"
            f"Inputs the writer was given:\n{_context(by_id[cid])}\n\n"
            f"Explanation to grade:\n{out['clinical_explanation']}\n\n"
            "Score 1-5 for faithfulness to the inputs and clarity (5 = every claim "
            "supported and clearly written). Set soft_hallucination=true if it makes "
            "any claim not supported by the inputs above."
        )
        result = _JUDGE.invoke(prompt)
        scores.append(result.score)
        soft.append(1 if result.soft_hallucination else 0)
    return {
        "judge_score_avg": round(mean(scores), 2) if scores else 0.0,
        "soft_hallucination_rate": round(mean(soft), 3) if soft else 0.0,
    }


def run(use_cache: bool = True, with_judge: bool = True) -> dict:
    cases = load_golden()
    outputs = load_or_generate(cases, use_cache=use_cache)
    metrics = score_deterministic(outputs)
    if with_judge:
        metrics.update(judge_outputs(cases, outputs))
    return metrics


def main() -> None:
    metrics = run()
    print("Node 5 (LLM explanation):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
