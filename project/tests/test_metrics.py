"""Retrieval metrics tests."""

from src.evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k():
    retrieved = [1, 2, 3, 4]
    relevant = {1, 3, 5}
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_recall_at_k():
    retrieved = [1, 2, 3]
    relevant = {1, 3, 5, 7}
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_mrr():
    assert mrr([4, 2, 1], {1, 5}) == 1 / 3


def test_ndcg_at_k():
    retrieved = [1, 2, 3]
    relevant = {1, 2}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0 < score <= 1.0
