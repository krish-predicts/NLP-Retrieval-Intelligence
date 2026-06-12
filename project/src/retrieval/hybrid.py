"""Hybrid BM25 + dense retriever."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.retrieval.base import RetrievalResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    min_v, max_v = values.min(), values.max()
    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class HybridRetriever:
    """Weighted combination of BM25 and dense scores."""

    method_name = "hybrid"

    def __init__(self, alpha: float | None = None, model_name: str | None = None) -> None:
        settings = get_settings()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=model_name)
        self.corpus: pd.DataFrame | None = None

    def fit(self, corpus: pd.DataFrame) -> None:
        self.corpus = corpus.reset_index(drop=True)
        self.bm25.fit(self.corpus)
        self.dense.fit(self.corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        bm25_results = self.bm25.search(query, top_k=top_k * 3, filters=filters)
        dense_results = self.dense.search(query, top_k=top_k * 3, filters=filters)

        bm25_scores = {r.review_id: r.score for r in bm25_results}
        dense_scores = {r.review_id: r.score for r in dense_results}
        bm25_norm = _min_max_normalize(bm25_scores)
        dense_norm = _min_max_normalize(dense_scores)

        all_ids = set(bm25_norm) | set(dense_norm)
        combined: list[tuple[int, float]] = []
        for rid in all_ids:
            score = self.alpha * bm25_norm.get(rid, 0.0) + (1 - self.alpha) * dense_norm.get(rid, 0.0)
            combined.append((rid, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        latency = (bm25_results[0].latency_ms if bm25_results else 0) + (
            dense_results[0].latency_ms if dense_results else 0
        )

        lookup = {r.review_id: r for r in bm25_results + dense_results}
        results: list[RetrievalResult] = []
        for i, (rid, score) in enumerate(combined[:top_k]):
            base = lookup[rid]
            results.append(
                RetrievalResult(
                    review_id=rid,
                    score=score,
                    text=base.text,
                    metadata=base.metadata,
                    latency_ms=latency if i == 0 else 0.0,
                    method=self.method_name,
                )
            )
        return results
