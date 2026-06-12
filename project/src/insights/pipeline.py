"""Insights pipeline orchestration."""

from __future__ import annotations

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.evaluation.business_metrics import compute_business_metrics
from src.insights.defects import mine_defects
from src.insights.feature_requests import mine_feature_requests
from src.insights.keyphrases import extract_keyphrases
from src.insights.sentiment import run_sentiment_analysis
from src.insights.stakeholder_reports import generate_stakeholder_report
from src.insights.topics import run_topic_modeling
from src.utils.io import save_json, save_parquet

logger = setup_logging(__name__)


def run_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Run full insight extraction pipeline."""
    topic_df, topic_summary = run_topic_modeling(df)
    sentiment_df = run_sentiment_analysis(topic_df)
    keyphrases = extract_keyphrases(sentiment_df)
    feature_df = mine_feature_requests(sentiment_df)
    defect_df = mine_defects(sentiment_df)
    generate_stakeholder_report(sentiment_df, topic_summary, feature_df, defect_df, keyphrases)

    metrics = compute_business_metrics(sentiment_df, feature_df, defect_df, sentiment_df)
    save_json(metrics, get_settings().reports / "business_metrics.json")

    settings = get_settings()
    save_parquet(sentiment_df, settings.processed)
    logger.info("Insights pipeline complete")
    return sentiment_df


def run_sentiment_stage(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run sentiment analysis only and persist enriched reviews."""
    settings = get_settings()
    if df is None:
        from src.utils.io import load_parquet

        df = load_parquet(settings.processed)

    sentiment_df = run_sentiment_analysis(df)
    save_parquet(sentiment_df, settings.processed)
    logger.info("Sentiment stage complete")
    return sentiment_df
