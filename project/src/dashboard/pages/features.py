"""Feature requests dashboard page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.charts import bar_chart
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import load_csv


def render() -> None:
    section_header("Feature Requests", "Ranked feature demand")
    features = load_csv("feature_requests.csv")
    if features.empty:
        st.info("Run insights pipeline to generate feature requests.")
        return

    fig = bar_chart(features.head(15), "sample_request", "count", "Top Feature Requests")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(features, use_container_width=True)
