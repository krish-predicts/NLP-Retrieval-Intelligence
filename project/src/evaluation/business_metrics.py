"""Business impact metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_business_metrics(
    df: pd.DataFrame,
    feature_hits: pd.DataFrame | None = None,
    defect_hits: pd.DataFrame | None = None,
    topic_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute stakeholder business coverage metrics."""
    low_rating = df[df["Score"] <= 2]
    complaint_coverage = 0.0
    if topic_df is not None and not topic_df.empty and len(low_rating) > 0:
        tagged = low_rating.merge(topic_df[["Id", "topic_id"]], on="Id", how="left")
        complaint_coverage = tagged["topic_id"].notna().mean()

    feature_coverage = 0.0
    if feature_hits is not None and len(df) > 0:
        feature_coverage = feature_hits["review_id"].nunique() / len(df)

    defect_rate = 0.0
    if defect_hits is not None and len(df) > 0:
        defect_rate = defect_hits["review_id"].nunique() / len(df)

    insight_coverage = 0.0
    if topic_df is not None and "sentiment" in topic_df.columns:
        insight_coverage = topic_df["topic_id"].notna().mean()

    return {
        "complaint_coverage": float(complaint_coverage),
        "feature_request_coverage": float(feature_coverage),
        "defect_detection_rate": float(defect_rate),
        "insight_generation_coverage": float(insight_coverage),
    }
