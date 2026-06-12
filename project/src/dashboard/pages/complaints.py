"""Complaints analysis dashboard page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.charts import bar_chart, line_chart
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import load_csv, load_reviews


def render() -> None:
    section_header("Complaints Analysis", "Trends and root causes")
    df = load_reviews()
    topics = load_csv("topic_summary.csv")
    if df.empty:
        st.warning("No data available.")
        return

    low = df[df["Score"] <= 2]
    trend = low.groupby("year_month").size().reset_index(name="complaints")
    fig = line_chart(trend, "year_month", "complaints", "Complaint Volume Over Time")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    if not topics.empty:
        complaints = topics.nsmallest(10, "avg_score")
        fig2 = bar_chart(complaints, "label", "count", "Top Complaint Topics")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(complaints, use_container_width=True)
