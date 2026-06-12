"""Cached data loaders for Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import get_settings


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    settings = get_settings()
    if settings.processed.exists():
        return pd.read_parquet(settings.processed)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    settings = get_settings()
    path = settings.reports / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json_report(name: str) -> dict:
    settings = get_settings()
    path = settings.reports / name
    if path.exists():
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def clear_dashboard_cache() -> None:
    """Clear Streamlit caches after pipeline runs."""
    load_reviews.clear()
    load_csv.clear()
    load_json_report.clear()


def artifact_missing_message(path: Path) -> str:
    return (
        f"Artifact not found: `{path}`. "
        "Run the pipeline: `python -m src.pipelines.run_all --stage sentiment`"
    )
