"""BM25 retriever using rank-bm25."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from src.retrieval.base import RetrievalResult
from src.retrieval.utils import apply_filters, row_to_result
from src.utils.timing import timer


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    """BM25Okapi retriever."""

    method_name = "bm25"

    def __init__(self) -> None:
        self.corpus: pd.DataFrame | None = None
        self.tokenized: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._index_map: list[int] = []

    def fit(self, corpus: pd.DataFrame) -> None:
        self.corpus = corpus.reset_index(drop=True)
        self.tokenized = [_tokenize(t) for t in self.corpus["combined_text"].astype(str)]
        self.bm25 = BM25Okapi(self.tokenized)
        self._index_map = list(range(len(self.corpus)))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self.corpus is None or self.bm25 is None:
            raise RuntimeError("Retriever not fitted")

        with timer() as elapsed:
            filtered = apply_filters(self.corpus, filters)
            if filtered.empty:
                return []

            query_tokens = _tokenize(query)
            scores = self.bm25.get_scores(query_tokens)
            filtered_scores = scores[filtered.index]
            top_positions = filtered_scores.argsort()[::-1][:top_k]

            results = [
                row_to_result(filtered.iloc[pos], filtered_scores[pos], 0.0, self.method_name)
                for pos in top_positions
            ]
        if results:
            results[0].latency_ms = elapsed[0]
        return results
