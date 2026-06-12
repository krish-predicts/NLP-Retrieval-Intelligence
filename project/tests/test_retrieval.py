"""Retriever tests."""

from pathlib import Path

import pandas as pd

from src.preprocessing.cleaner import clean_reviews
from src.preprocessing.metadata import add_metadata
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.registry import get_retriever
from src.retrieval.tfidf import TfidfRetriever


def _sample_corpus() -> pd.DataFrame:
    fixture = Path(__file__).parent / "fixtures" / "sample_reviews.csv"
    df = pd.read_csv(fixture)
    df = clean_reviews(df)
    return add_metadata(df)


def test_tfidf_retriever_returns_top_k():
    corpus = _sample_corpus()
    retriever = TfidfRetriever()
    retriever.fit(corpus)
    results = retriever.search("shipping problems", top_k=3)
    assert len(results) <= 3
    assert results[0].latency_ms >= 0
    assert results[0].method == "tfidf"


def test_bm25_retriever_returns_results():
    corpus = _sample_corpus()
    retriever = BM25Retriever()
    retriever.fit(corpus)
    results = retriever.search("defective broken", top_k=5)
    assert len(results) > 0


def test_registry_get_retriever():
    retriever = get_retriever("tfidf")
    assert retriever.method_name == "tfidf"
