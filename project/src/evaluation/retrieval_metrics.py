"""Retrieval evaluation metrics."""

from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Compute Recall@K."""
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for doc in top if doc in relevant) / len(relevant)


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(top) if doc in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def latency_stats(latencies: Iterable[float]) -> dict[str, float]:
    """Compute p50 and p95 latency."""
    values = sorted(latencies)
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}

    def percentile(p: float) -> float:
        idx = int(len(values) * p)
        idx = min(idx, len(values) - 1)
        return values[idx]

    return {
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "mean_ms": sum(values) / len(values),
    }
