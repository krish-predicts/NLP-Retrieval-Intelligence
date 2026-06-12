"""Retriever factory registry."""

from __future__ import annotations

from typing import Literal

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.tfidf import TfidfRetriever

RetrieverName = Literal["tfidf", "bm25", "dense", "hybrid"]


def get_retriever(name: RetrieverName, model_name: str | None = None) -> BaseRetriever:
    """Return retriever instance by name."""
    if name == "tfidf":
        return TfidfRetriever()
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever(model_name=model_name)
    if name == "hybrid":
        return HybridRetriever(model_name=model_name)
    raise ValueError(f"Unknown retriever: {name}")
