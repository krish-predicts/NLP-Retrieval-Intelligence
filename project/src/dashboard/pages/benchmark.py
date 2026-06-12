"""Retrieval benchmark dashboard page."""

from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.dashboard.components.charts import bar_chart
from src.dashboard.components.kpi import section_header
from src.dashboard.data_loader import artifact_missing_message, clear_dashboard_cache, load_csv, load_reviews


def _render_benchmark_charts(benchmark) -> None:
    metric_cols = [c for c in benchmark.columns if "precision" in c or c in ("mrr", "latency_ms")]
    agg = benchmark.groupby("retriever")[metric_cols].mean().reset_index()

    if "latency_ms" in agg.columns:
        fig = bar_chart(agg, "retriever", "latency_ms", "Mean Query Latency (ms)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    prec_col = next((c for c in agg.columns if "precision" in c), None)
    if prec_col:
        fig2 = bar_chart(agg, "retriever", prec_col, f"Mean {prec_col}")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(benchmark, use_container_width=True)


def render() -> None:
    section_header("Retrieval Benchmark", "Model comparison and latency")
    settings = get_settings()

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run Retrieval Benchmark", type="primary", use_container_width=True):
            df = load_reviews()
            if df.empty:
                st.error(artifact_missing_message(settings.processed))
            else:
                with st.spinner("Evaluating TF-IDF, BM25, Dense, and Hybrid retrievers..."):
                    from src.evaluation.experiment_runner import run_full_benchmark

                    run_full_benchmark(df)
                    clear_dashboard_cache()
                st.success("Retrieval benchmark complete.")
                st.rerun()

    benchmark = load_csv("retrieval_benchmark.csv")
    if benchmark.empty:
        st.info(
            'Click **Run Retrieval Benchmark** to generate metrics, or run: '
            '`python -m src.pipelines.run_all --stage evaluation`'
        )
        return

    _render_benchmark_charts(benchmark)
