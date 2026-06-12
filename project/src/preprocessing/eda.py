"""Exploratory data analysis summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.utils.io import save_json

logger = setup_logging(__name__)


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Compute EDA statistics and persist to reports."""
    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)

    rating_dist = df["Score"].value_counts().sort_index().to_dict()
    volume_trends = df.groupby("year_month").size().to_dict()
    product_stats = (
        df.groupby("ProductId")
        .agg(review_count=("Id", "count"), avg_score=("Score", "mean"))
        .sort_values("review_count", ascending=False)
        .head(20)
        .reset_index()
        .to_dict(orient="records")
    )
    length_stats = {
        "mean_length": float(df["review_length"].mean()),
        "median_length": float(df["review_length"].median()),
        "mean_word_count": float(df["word_count"].mean()),
        "median_word_count": float(df["word_count"].median()),
    }

    summary: dict[str, Any] = {
        "total_reviews": len(df),
        "rating_distribution": rating_dist,
        "avg_rating": float(df["Score"].mean()),
        "volume_trends": volume_trends,
        "top_products": product_stats,
        "length_statistics": length_stats,
    }

    save_json(summary, settings.reports / "eda_summary.json")
    logger.info("EDA summary saved to reports/eda_summary.json")
    return summary
