"""Dense embedding retriever with FAISS."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import get_settings
from src.embeddings.faiss_index import FaissIndex
from src.retrieval.base import RetrievalResult
from src.retrieval.utils import row_to_result
from src.utils.timing import timer


class DenseRetriever:
    """Dense retrieval via SentenceTransformers + FAISS."""

    method_name = "dense"

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.default_embedding_model
        self.faiss_index = FaissIndex(self.model_name)
        self.corpus: pd.DataFrame | None = None

    def fit(self, corpus: pd.DataFrame) -> None:
        self.corpus = corpus.reset_index(drop=True)
        if self.faiss_index.index_path.exists():
            self.faiss_index.load()
        else:
            self.faiss_index.build(self.corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        with timer() as elapsed:
            hits = self.faiss_index.search(query, top_k=top_k, filters=filters)
            results = [
                row_to_result(row, score, 0.0, self.method_name)
                for _, score, row in hits
            ]
        if results:
            results[0].latency_ms = elapsed[0]
        return results
