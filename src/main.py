"""Smoke test / CLI runner.

No args  -> runs 3 hardcoded samples.
With args -> runs a single case from the command line.
"""

import argparse

from src.graph import build_graph
from src.state import AgentState, UserData


SAMPLES: list[tuple[str, AgentState]] = [
    (
        "Healthy adult, clean image",
        AgentState(
            video_id="video_normal",
            user_data=UserData(age=30, symptoms=[], prior_history=[]),
        ),
    ),
    (
        "1-year-old with high fever, AOM image",
        AgentState(
            video_id="video_aom",
            user_data=UserData(age=1, symptoms=["high_fever", "ear_pain"], prior_history=[]),
        ),
    ),
    (
        "Blurry image, should re_capture",
        AgentState(
            video_id="video_blurry",
            user_data=UserData(age=5, symptoms=[], prior_history=[]),
        ),
    ),
]


def _print_result(label: str, result: AgentState) -> None:
    print(f"\n=== {label} ===")
    print(f"decision         : {result.decision}")
    if result.recapture_reason:
        print(f"recapture_reason : {result.recapture_reason}")
    if result.risk_score is not None:
        print(f"risk_score       : {result.risk_score:.2f}")
    if result.explanation:
        print(f"explanation      : {result.explanation}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the triage agent on a single case.")
    p.add_argument("--video", help="Mock video id: video_normal | video_aom | video_blurry")
    p.add_argument("--age", type=int, help="Patient age")
    p.add_argument("--symptoms", nargs="*", default=[], help="Symptoms, space-separated")
    p.add_argument("--history", nargs="*", default=[], help="Prior history, space-separated")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    graph = build_graph()

    if args.video and args.age is not None:
        sample = AgentState(
            video_id=args.video,
            user_data=UserData(age=args.age, symptoms=args.symptoms, prior_history=args.history),
        )
        result = AgentState(**graph.invoke(sample))
        _print_result("CLI case", result)
        return

    for label, sample in SAMPLES:
        result = AgentState(**graph.invoke(sample))
        _print_result(label, result)


if __name__ == "__main__":
    main()
