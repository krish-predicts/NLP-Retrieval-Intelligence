"""Defect detection dashboard page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.charts import bar_chart
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import load_csv


def render() -> None:
    section_header("Defect Detection", "Defect frequency and risk alerts")
    defects = load_csv("defect_report.csv")
    if defects.empty:
        st.info("Defect report not available.")
        return

    fig = bar_chart(defects.head(15), "defect_type", "count", "Defect Frequency")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    if "immediate_action" in defects.columns:
        alerts = defects[defects["immediate_action"] == True]  # noqa: E712
        if not alerts.empty:
            st.error("Immediate action required for:")
            st.dataframe(alerts, use_container_width=True)

    st.dataframe(defects, use_container_width=True)
