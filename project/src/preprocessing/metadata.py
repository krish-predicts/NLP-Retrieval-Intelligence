"""Review metadata generation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metadata fields for analytics and filtering."""
    work = df.copy()
    work["combined_text"] = (
        work["Summary"].fillna("").astype(str) + " " + work["Text"].astype(str)
    ).str.strip()
    work["review_length"] = work["combined_text"].str.len()
    work["word_count"] = work["combined_text"].str.split().str.len()
    work["review_date"] = pd.to_datetime(work["Time"], unit="s", utc=True)
    work["year_month"] = work["review_date"].dt.strftime("%Y-%m")
    work["helpfulness_ratio"] = work.apply(
        lambda row: (
            row["HelpfulnessNumerator"] / row["HelpfulnessDenominator"]
            if row["HelpfulnessDenominator"] > 0
            else 0.0
        ),
        axis=1,
    )
    return work
