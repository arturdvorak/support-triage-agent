# Evaluation Report

Date: 2026-06-17
Golden set: 40 cases. Small sample, so numbers are directional with wide error bars.

## Headline

- End-to-end sensitivity: 1.0 (under-triage 0.0)
- End-to-end specificity: 0.857 (over-triage 0.143)
- Node 5 hard hallucination rate: 0.216216 (target 0)

## End-to-end decision

- macro-F1: 0.932
- Plain accuracy is intentionally not the lead: under class imbalance it flatters a majority-class guesser.

## Retrieval (Node 4)

- precision@3: 1.0
- recall@3: 0.12
- mrr: 1.0
- ndcg@3: 0.6

## Node 5 (LLM explanation)

- schema pass rate: 1.0
- hard hallucination rate: 0.216216
- soft hallucination rate: 1
- judge score (1-5 avg): 2.84

