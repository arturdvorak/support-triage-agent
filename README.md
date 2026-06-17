# Triage Agent

A 6-step LangGraph workflow that turns a CNN's ear-image prediction into a plain-language explanation plus a clear next-step decision (`auto_clear`, `escalate_*`, or `re_capture`).

---

## Problem and Solution

**Problem.** A mobile app captures a video of a patient's eardrum. A CNN can label the image, but a raw label (e.g. `aom, 0.88`) is not useful to the patient and is hard to audit in a regulated healthcare setting.

**Solution.** The triage agent takes the CNN output plus patient context (age, symptoms, history), looks up similar past cases, asks an LLM to write a plain-language explanation grounded in those cases, and applies deterministic rules to produce one final decision. Every step's input and output is saved so the decision can be replayed for audit.

**Key split:**

- The **CNN owns the diagnosis** (predictable, validated, the only thing approved to call a finding).
- The **LLM only explains and routes** (no clinical claims, no hallucinated diagnoses).

---

## Architecture Overview

The agent runs as a fixed 6-node graph. Node 2 can short-circuit to `re_capture`; otherwise the flow runs through all 6 nodes.


| Node | Name                      | What it does                                                                                 |
| ---- | ------------------------- | -------------------------------------------------------------------------------------------- |
| 1    | CNN Inference             | Calls the CNN once; stores visibility, quality flags, diagnosis, cnn_confidences             |
| 2    | Quality + Visibility Gate | If eardrum not fully visible or any quality flag fired, set decision = `re_capture` and stop |
| 3    | Risk Scoring              | Combine diagnosis with age, symptoms, prior history into a 0-1 risk score                    |
| 4    | Similar Case Retrieval    | Look up past cases with the same diagnosis (real ChromaDB over 100 synthetic cases)          |
| 5    | LLM Explanation           | Claude writes a 2-3 sentence explanation referencing the retrieved case IDs                  |
| 6    | Escalation Decision       | Rule-based check over 3 signals (CNN, retrieved cases, LLM uncertainty); outputs final label |


End-to-end flow (production target):

```
Mobile app
    | (HTTPS, video + metadata)
    v
FastAPI + Pydantic   <- input validation, auth
    |
    v
LangGraph agent
    | Node 1: CNN Inference          (one call, 3 outputs)
    | Node 2: Quality + Visibility   (may exit to re_capture)
    | Node 3: Risk Scoring
    | Node 4: Similar Case Retrieval (ChromaDB)
    | Node 5: LLM Explanation        (Claude via LangChain)
    | Node 6: Escalation Decision    (rules)
    v
Final decision + explanation -> mobile app
                          \-> PostgreSQL (audit log)
                          \-> LangSmith (trace)
```

The full diagram with cloud / HIPAA boundaries and the decision table for Node 6 lives in `[interview-story/q2_session2_technical_approach.md](interview-story/q2_session2_technical_approach.md)`.

---

## Tech Stack


| Layer         | Technology                         | Status in this repo               |
| ------------- | ---------------------------------- | --------------------------------- |
| API           | FastAPI + Pydantic                 | Not implemented (CLI runner only) |
| Orchestration | LangGraph                          | Implemented                       |
| Diagnosis     | EfficientNetV2 multi-task CNN      | Mocked (`src/mock_services.py`)   |
| LLM           | Claude (via `langchain-anthropic`) | Real                              |
| Vector store  | ChromaDB                           | Implemented (local persistent)    |
| Audit DB      | PostgreSQL                         | Not implemented                   |
| Observability | LangSmith                          | Implemented (auto-tracing)        |
| Deployment    | Docker + Kubernetes                | Not implemented                   |


The prototype focuses on the agent loop. Everything outside the agent (API, DB, deployment) is described in the Q2 doc but not built here.

---

## Project Structure

```
triage-agent/
  src/
    main.py            # CLI runner: 3 hardcoded samples or one --video case
    graph.py           # LangGraph wiring of the 6 nodes
    state.py           # Pydantic AgentState (shared across all nodes)
    nodes.py           # The 6 node functions + LLM prompt
    mock_services.py   # Fake CNN (only)
    retrieval.py       # Real ChromaDB case retrieval (Node 4)
  data/
    generate_cases.py  # Deterministic generator for the synthetic cases
    cases.json         # 100 synthetic past cases (committed)
  interview-story/     # Q1 (business) and Q2 (technical) write-ups
  requirements.txt
  .env.example         # ANTHROPIC_API_KEY placeholder
```

---

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then open .env and paste your real ANTHROPIC_API_KEY
```

First run note: Node 4 uses a local sentence-transformers model
(`all-MiniLM-L6-v2`, about 80MB). It downloads once on the first run and caches
under `~/.cache/huggingface/`. The first run also seeds a local ChromaDB store
in `chroma_db/` from `data/cases.json`. Both are one-time; later runs are fast
and offline. To regenerate the cases, run `python -m data.generate_cases`.

---

## Observability (LangSmith)

LangSmith tracing is wired in but optional. If `LANGSMITH_API_KEY` is missing, tracing is silently off and the app runs unchanged.

To turn it on, add these to your `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT=triage-agent-dev
```

Then each `python -m src.main` run shows up at `https://smith.langchain.com` under the project name, with:

- One top-level span per case, named after its label (e.g. `Healthy adult, clean image`).
- 6 child spans, one per node. The Claude call is nested inside `node5_llm` with full prompt, response, tokens, latency, and cost.
- Each run tagged with `video_id` and carrying `age` + `symptoms` as metadata, so you can filter runs in the LangSmith UI.

LangSmith is observability only - it is not the system of record. In production, decisions still get written to PostgreSQL for audit.

---

## How to Run

**Default: run all 3 hardcoded samples** (healthy adult, infant with AOM, blurry image).

```bash
python -m src.main
```

**Custom: run a single case from the CLI.**

```bash
python -m src.main --video video_aom --age 1 --symptoms high_fever ear_pain
```

Available mock `--video` values: `video_normal`, `video_aom`, `video_blurry`.

Each run prints the final `decision`, the `risk_score`, the LLM `clinical_explanation`, and (for `re_capture` cases) the user-facing `recapture_reason`.

---

## Run the API

Start the server:

```bash
uvicorn src.api:app --reload
```

Then call the triage endpoint:

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"video_id": "video_aom", "age": 1, "symptoms": ["high_fever", "ear_pain"]}'
```

The response has two blocks: `patient` (decision, fixed safety message, plain-language
explanation) and `clinical` (risk score, confidence, similar case IDs, clinical
explanation). Interactive docs are at `http://127.0.0.1:8000/docs`.

---

## What Is Mocked vs Production


| Component       | This repo                                               | Production target                                      |
| --------------- | ------------------------------------------------------- | ------------------------------------------------------ |
| CNN inference   | `mock_cnn()` returns canned output per `video_id`       | Real EfficientNetV2 served behind an internal endpoint |
| Case retrieval  | Real ChromaDB over 100 synthetic cases (text embeddings) | ChromaDB lookup by image embedding over real cases    |
| LLM explanation | Real Claude call via `langchain-anthropic`              | Same                                                   |
| API             | None (CLI only)                                         | FastAPI + Pydantic with auth and input validation      |
| Persistence     | None (in-memory state)                                  | PostgreSQL for decisions, LangSmith for traces         |


The mocks keep the same input/output shape as the real services, so swapping them in later means changing the function body only - not the call sites.

---

## Evaluation

A runnable evaluation layer scores three parts of the agent against a 40-case
golden dataset (`data/golden.json`):

- Retrieval (Node 4): precision@3, recall@3, MRR, NDCG@3.
- Node 5 (LLM explanation): schema pass rate, hard and soft hallucination rate,
  LLM-judge score.
- End-to-end decision: sensitivity and specificity on the escalate-vs-clear binary
  (with under-triage and over-triage), plus macro-F1.

Build the golden set once, then run all evaluations:

```bash
python -m data.build_golden_dataset
python -m evaluation.run_all
```

`run_all` writes `evaluation/report.md`. Node 5 generations are cached in
`evaluation/node5_outputs.json`; delete it to regenerate after a prompt or model
change.