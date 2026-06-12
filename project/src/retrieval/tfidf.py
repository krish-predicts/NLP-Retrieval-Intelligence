"""TF-IDF cosine similarity retriever."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.base import RetrievalResult
from src.retrieval.utils import apply_filters, row_to_result
from src.utils.timing import timer


class TfidfRetriever:
    """TF-IDF + cosine similarity retrieval."""

    method_name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.corpus: pd.DataFrame | None = None
        self.matrix = None

    def fit(self, corpus: pd.DataFrame) -> None:
        self.corpus = corpus.reset_index(drop=True)
        texts = self.corpus["combined_text"].astype(str).tolist()
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self.corpus is None or self.matrix is None:
            raise RuntimeError("Retriever not fitted")

        with timer() as elapsed:
            filtered = apply_filters(self.corpus, filters)
            if filtered.empty:
                return []

            indices = filtered.index.tolist()
            sub_matrix = self.matrix[indices]
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, sub_matrix).flatten()
            top_indices = scores.argsort()[::-1][:top_k]

            results = [
                row_to_result(filtered.iloc[pos], scores[top_indices[i]], 0.0, self.method_name)
                for i, pos in enumerate(top_indices)
            ]
        if results:
            results[0].latency_ms = elapsed[0]
        return results
