"""Preprocessing pipeline orchestration."""

from __future__ import annotations

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.preprocessing.cleaner import clean_reviews
from src.preprocessing.eda import run_eda
from src.preprocessing.metadata import add_metadata
from src.utils.io import load_parquet, save_parquet

logger = setup_logging(__name__)


def run_preprocessing(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean text, add metadata, run EDA, and persist processed corpus."""
    settings = get_settings()
    if df is None:
        df = load_parquet(settings.processed_raw)

    cleaned = clean_reviews(df)
    enriched = add_metadata(cleaned)
    run_eda(enriched)

    settings.processed.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(enriched, settings.processed)
    logger.info("Preprocessing complete: %d rows", len(enriched))
    return enriched
