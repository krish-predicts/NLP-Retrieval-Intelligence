"""Pseudo relevance judgments for evaluation."""

from __future__ import annotations

import pandas as pd

from src.evaluation.queries import EVAL_QUERIES
from src.retrieval.bm25 import BM25Retriever


def build_pseudo_qrels(corpus: pd.DataFrame, query: str, must_match: list[str] | None = None) -> set[int]:
    """Build pseudo-relevant doc IDs using phrase match + BM25 ranking."""
    terms = must_match or EVAL_QUERIES.get(query, query.lower().split())
    text = corpus["combined_text"].astype(str).str.lower()
    mask = pd.Series(True, index=corpus.index)
    for term in terms:
        mask &= text.str.contains(term, regex=False, na=False)
    candidates = corpus[mask]
    if candidates.empty:
        bm25 = BM25Retriever()
        bm25.fit(corpus)
        results = bm25.search(query, top_k=20)
        return {r.review_id for r in results}

    bm25 = BM25Retriever()
    bm25.fit(candidates)
    results = bm25.search(query, top_k=min(20, len(candidates)))
    return {r.review_id for r in results}
