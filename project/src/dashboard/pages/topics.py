"""Topic explorer dashboard page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.charts import bar_chart, scatter_chart
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import load_csv, load_reviews


def render() -> None:
    section_header("Topic Explorer", "Topic clusters and evolution")
    topics = load_csv("topic_summary.csv")
    df = load_reviews()
    if topics.empty:
        st.info("Topic summary not available.")
        return

    fig = scatter_chart(topics, "count", "avg_score", "label", "Topic Size vs Avg Rating")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    fig2 = bar_chart(topics.head(15), "label", "count", "Top Topics by Volume")
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

    if not df.empty and "topic_id" in df.columns:
        evolution = df.groupby(["year_month", "topic_id"]).size().reset_index(name="count")
        st.dataframe(evolution.head(50), use_container_width=True)
