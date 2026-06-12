"""Retrieval search dashboard page."""

from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import load_reviews
from src.retrieval.registry import get_retriever


@st.cache_resource(show_spinner=True)
def _get_fitted_retriever(name: str):
    df = load_reviews()
    retriever = get_retriever(name)  # type: ignore[arg-type]
    retriever.fit(df)
    return retriever


def render() -> None:
    section_header("Retrieval Search", "Natural language search across reviews")
    df = load_reviews()
    if df.empty:
        st.warning("No processed reviews found.")
        return

    settings = get_settings()
    query = st.text_input("Query", value="shipping problems")
    retriever_name = st.selectbox("Retriever", ["tfidf", "bm25", "dense", "hybrid"])
    top_k = st.slider("Top K", 5, 20, settings.top_k)
    min_score = st.slider("Min Rating Filter", 1, 5, 1)
    filters = {"min_score": min_score} if min_score > 1 else None

    if st.button("Search"):
        retriever = _get_fitted_retriever(retriever_name)
        results = retriever.search(query, top_k=top_k, filters=filters)
        for i, result in enumerate(results, start=1):
            st.markdown(f"**#{i}** | Score: {result.score:.4f} | Rating: {result.metadata.get('Score')}")
            st.write(result.text[:500])
            st.caption(f"Latency: {result.latency_ms:.1f} ms | Method: {result.method}")
