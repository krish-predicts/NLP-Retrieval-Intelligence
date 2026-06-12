"""Base retriever protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class RetrievalResult:
    """Single retrieval hit."""

    review_id: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    method: str = ""


class BaseRetriever(Protocol):
    """Retriever interface."""

    method_name: str

    def fit(self, corpus: pd.DataFrame) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...
