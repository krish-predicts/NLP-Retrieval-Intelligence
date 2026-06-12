"""Ingestion pipeline orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.ingestion.deduplicator import deduplicate_reviews
from src.ingestion.loader import load_reviews
from src.ingestion.sampler import stratified_sample
from src.ingestion.validator import validate_reviews
from src.utils.io import save_json, save_parquet

logger = setup_logging(__name__)


def run_ingestion() -> pd.DataFrame:
    """Load, validate, dedupe, sample, and persist raw processed data."""
    settings = get_settings()
    settings.processed_raw.parent.mkdir(parents=True, exist_ok=True)
    settings.reports.mkdir(parents=True, exist_ok=True)

    df = load_reviews(settings.raw_data)
    validated, validation_report = validate_reviews(df)
    deduped = deduplicate_reviews(validated)
    sampled = stratified_sample(deduped)

    save_parquet(sampled, settings.processed_raw)
    save_json(validation_report, settings.reports / "validation_report.json")

    logger.info("Ingestion complete: %d rows saved", len(sampled))
    return sampled
