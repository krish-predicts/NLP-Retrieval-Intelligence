"""Sentiment analysis dashboard page."""

from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.charts import bar_chart
from src.dashboard.components.kpi import kpi_card, section_header
from src.dashboard.data_loader import artifact_missing_message, clear_dashboard_cache, load_reviews


def _render_sentiment_charts(df) -> None:
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig = bar_chart(sentiment_counts, "sentiment", "count", "Overall Sentiment Distribution")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    if "sentiment_aligned" in df.columns:
        alignment = df["sentiment_aligned"].mean() * 100
        kpi_card("Rating-Sentiment Alignment", f"{alignment:.1f}%")

    product_sent = (
        df.groupby("ProductId")
        .agg(avg_score=("Score", "mean"), review_count=("Id", "count"))
        .reset_index()
        .sort_values("review_count", ascending=False)
        .head(20)
    )
    st.dataframe(product_sent, use_container_width=True)


def render() -> None:
    section_header("Sentiment Analysis", "Product and rating sentiment")
    settings = get_settings()
    df = load_reviews()

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run Sentiment Analysis", type="primary", use_container_width=True):
            if df.empty:
                st.error(artifact_missing_message(settings.processed))
            else:
                with st.spinner("Training sentiment model and scoring reviews..."):
                    from src.insights.pipeline import run_sentiment_stage

                    run_sentiment_stage(df.copy())
                    clear_dashboard_cache()
                st.success("Sentiment analysis complete.")
                st.rerun()

    df = load_reviews()
    if df.empty:
        st.warning("No processed reviews found. Run ingestion and preprocessing first.")
        return

    if "sentiment" not in df.columns:
        st.info('Click **Run Sentiment Analysis** to generate sentiment columns, or run: '
                '`python -m src.pipelines.run_all --stage sentiment`')
        return

    _render_sentiment_charts(df)
