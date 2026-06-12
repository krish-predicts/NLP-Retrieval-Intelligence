"""Text cleaning utilities."""

from __future__ import annotations

import re

import pandas as pd

from src.config.settings import get_settings


def clean_text(text: str, lowercase: bool | None = None) -> str:
    """Clean review text while preserving sentiment-bearing terms."""
    settings = get_settings()
    use_lower = lowercase if lowercase is not None else settings.lowercase_text

    cleaned = str(text)
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-\']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if use_lower:
        cleaned = cleaned.lower()
    return cleaned


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to Summary and Text columns."""
    work = df.copy()
    work["Summary"] = work["Summary"].fillna("").astype(str).map(clean_text)
    work["Text"] = work["Text"].astype(str).map(clean_text)
    return work
