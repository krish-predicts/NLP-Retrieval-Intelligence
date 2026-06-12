"""Deduplicate reviews."""

from __future__ import annotations

import re

import pandas as pd

from src.config.logging_config import setup_logging

logger = setup_logging(__name__)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    return cleaned


def deduplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe on UserId + ProductId + normalized Text; keep latest Time."""
    work = df.copy()
    work["_norm_text"] = work["Text"].astype(str).map(_normalize_text)
    work = work.sort_values("Time", ascending=False)
    before = len(work)
    deduped = work.drop_duplicates(subset=["UserId", "ProductId", "_norm_text"], keep="first")
    deduped = deduped.drop(columns=["_norm_text"]).reset_index(drop=True)
    logger.info("Deduplication removed %d rows", before - len(deduped))
    return deduped
