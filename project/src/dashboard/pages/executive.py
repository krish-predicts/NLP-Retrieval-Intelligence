"""Executive overview dashboard page."""

from __future__ import annotations

import streamlit as st

from src.dashboard.components.kpi import kpi_card, section_header
from src.dashboard.data_loader import load_csv, load_json_report, load_reviews


def render() -> None:
    section_header("Executive Overview", "KPIs and top findings for leadership")
    df = load_reviews()
    eda = load_json_report("eda_summary.json")
    defects = load_csv("defect_report.csv")

    if df.empty:
        st.warning("Run the pipeline to generate data artifacts.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Reviews", f"{len(df):,}")
    with col2:
        kpi_card("Avg Rating", f"{df['Score'].mean():.2f}")
    with col3:
        kpi_card("Products", f"{df['ProductId'].nunique():,}")
    with col4:
        low_pct = (df["Score"] <= 2).mean() * 100
        kpi_card("Low Ratings", f"{low_pct:.1f}%")

    st.markdown("### Top Findings")
    if eda:
        st.write(f"Rating distribution: {eda.get('rating_distribution', {})}")

    st.markdown("### Emerging Issues")
    if not defects.empty:
        st.dataframe(defects.head(10), use_container_width=True)
    else:
        st.info("No defect report available yet.")
