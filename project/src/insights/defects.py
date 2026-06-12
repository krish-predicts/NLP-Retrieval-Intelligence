"""Defect pattern mining."""

from __future__ import annotations

import re

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)

DEFECT_PATTERNS = [
    "broken",
    "defective",
    "stopped working",
    "cracked",
    "overheating",
    "malfunction",
    "spoiled",
    "mold",
    "rancid",
    "stale",
    "expired",
]


def mine_defects(df: pd.DataFrame) -> pd.DataFrame:
    """Detect defect mentions and flag escalation priorities."""
    settings = get_settings()
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        text = str(row["combined_text"]).lower()
        for pattern in DEFECT_PATTERNS:
            if pattern in text:
                records.append(
                    {
                        "review_id": row["Id"],
                        "ProductId": row["ProductId"],
                        "Score": row["Score"],
                        "year_month": row.get("year_month"),
                        "defect_type": pattern,
                    }
                )

    if not records:
        empty = pd.DataFrame(columns=["defect_type", "count", "avg_score", "immediate_action"])
        empty.to_csv(settings.reports / "defect_report.csv", index=False)
        return empty

    defects = pd.DataFrame(records)
    recent_months = sorted(df["year_month"].dropna().unique())[-3:]
    summary = (
        defects.groupby("defect_type")
        .agg(
            count=("review_id", "count"),
            avg_score=("Score", "mean"),
            recent_count=("year_month", lambda s: int((s.isin(recent_months)).sum())),
        )
        .reset_index()
    )
    summary["immediate_action"] = summary["recent_count"] >= summary["count"] * 0.4
    summary = summary.sort_values("count", ascending=False)
    summary.to_csv(settings.reports / "defect_report.csv", index=False)
    logger.info("Defect mining complete: %d mentions", len(defects))
    return defects
