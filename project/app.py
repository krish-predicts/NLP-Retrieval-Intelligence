"""NLP Retrieval Intelligence Platform — Streamlit Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.pages import benchmark, complaints, defects, executive, features, search, sentiment, topics
from src.utils.multiprocessing_guard import configure_multiprocessing

PAGES = {
    "Executive Overview": executive.render,
    "Retrieval Search": search.render,
    "Complaints Analysis": complaints.render,
    "Feature Requests": features.render,
    "Sentiment Analysis": sentiment.render,
    "Topic Explorer": topics.render,
    "Defect Detection": defects.render,
    "Retrieval Benchmark": benchmark.render,
}


def main() -> None:
    configure_multiprocessing()

    st.set_page_config(
        page_title="NLP Retrieval Intelligence",
        page_icon="🔍",
        layout="wide",
    )
    st.title("NLP Retrieval Intelligence Platform")
    st.caption("Amazon Reviews — Retrieval, Insights, and Business Analytics")

    st.sidebar.markdown("### Navigation")
    selection = st.sidebar.radio("Pages", list(PAGES.keys()), label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick actions**")
    st.sidebar.caption("Use Sentiment Analysis or Retrieval Benchmark pages to run pipelines from the UI.")

    PAGES[selection]()


if __name__ == "__main__":
    main()
