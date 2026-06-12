"""Shared retrieval utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.retrieval.base import RetrievalResult


def apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Filter corpus by metadata fields."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key == "min_score" and "Score" in df.columns:
            mask &= df["Score"] >= int(value)
        elif key in df.columns:
            mask &= df[key] == value
    return df[mask]


def row_to_result(
    row: pd.Series,
    score: float,
    latency_ms: float,
    method: str,
    text_col: str = "combined_text",
) -> RetrievalResult:
    """Convert dataframe row to RetrievalResult."""
    return RetrievalResult(
        review_id=int(row["Id"]),
        score=float(score),
        text=str(row.get(text_col, "")),
        metadata={
            "Score": row.get("Score"),
            "ProductId": row.get("ProductId"),
            "year_month": row.get("year_month"),
            "Summary": row.get("Summary"),
        },
        latency_ms=latency_ms,
        method=method,
    )
