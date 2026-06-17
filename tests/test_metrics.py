"""Unit tests for the pure metric helpers."""

import math

import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)
from evaluation.metrics import macro_f1, sensitivity_specificity

# relevance: case_id -> grade (2 = strong, 1 = weak). Missing id means grade 0.
REL = {"c1": 2, "c2": 1, "c3": 2}


def test_precision_at_k_strong_only():
    # top-3 = c1(2), c2(1), c9(0). Strong (grade>=2) hits: c1 -> 1/3.
    assert precision_at_k(["c1", "c2", "c9"], REL, k=3, rel=2) == pytest.approx(1 / 3)


def test_recall_at_k_strong_only():
    # strong relevant set = {c1, c3} (size 2). top-3 contains c1 only -> 1/2.
    assert recall_at_k(["c1", "c2", "c9"], REL, k=3, rel=2) == pytest.approx(0.5)


def test_reciprocal_rank_first_strong():
    # first strong (grade>=2) is c3 at rank 2 -> 0.5.
    assert reciprocal_rank(["c2", "c3", "c1"], REL, rel=2) == pytest.approx(0.5)


def test_reciprocal_rank_none_relevant():
    assert reciprocal_rank(["c9", "c8"], REL, rel=2) == 0.0


def test_ndcg_perfect_order_is_one():
    # graded gains, ideal order already -> 1.0.
    assert ndcg_at_k(["c1", "c3", "c2"], REL, k=3) == pytest.approx(1.0)


def test_ndcg_rewards_strong_first():
    # weak before strong scores lower than strong before weak.
    worse = ndcg_at_k(["c2", "c1"], REL, k=2)
    better = ndcg_at_k(["c1", "c2"], REL, k=2)
    assert better > worse
    # exact check on the worse ordering: grades [1, 2].
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    assert worse == pytest.approx(dcg / idcg)


def test_macro_f1_simple():
    expected = ["a", "a", "b"]
    predicted = ["a", "b", "b"]
    # class a: tp=1, fp=0, fn=1 -> f1 = 0.6667
    # class b: tp=1, fp=1, fn=0 -> f1 = 0.6667
    assert macro_f1(expected, predicted) == pytest.approx(2 / 3)


def test_sensitivity_specificity_collapse():
    expected = ["escalate_urgent", "auto_clear", "auto_clear", "re_capture"]
    predicted = ["escalate_urgent", "escalate_routine", "auto_clear", "re_capture"]
    out = sensitivity_specificity(expected, predicted)
    # escalate rows: 1 case, caught -> sensitivity 1.0
    assert out["sensitivity"] == pytest.approx(1.0)
    # clear rows: 2 cases, 1 wrongly escalated -> specificity 0.5
    assert out["specificity"] == pytest.approx(0.5)
    assert out["under_triage"] == pytest.approx(0.0)
    assert out["over_triage"] == pytest.approx(0.5)
