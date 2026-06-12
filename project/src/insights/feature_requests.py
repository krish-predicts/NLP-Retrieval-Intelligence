"""Feature request mining and clustering."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)

FEATURE_PATTERNS = [
    r"wish it had",
    r"should include",
    r"would like",
    r"\bneeds\b",
    r"\bmissing\b",
]


def _extract_feature_spans(text: str) -> list[str]:
    spans: list[str] = []
    lower = text.lower()
    for pattern in FEATURE_PATTERNS:
        for match in re.finditer(pattern, lower):
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 80)
            spans.append(text[start:end].strip())
    return spans


def mine_feature_requests(df: pd.DataFrame, max_clusters: int = 8) -> pd.DataFrame:
    """Detect and cluster feature request patterns."""
    settings = get_settings()
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        spans = _extract_feature_spans(str(row["combined_text"]))
        for span in spans:
            records.append(
                {
                    "review_id": row["Id"],
                    "ProductId": row["ProductId"],
                    "Score": row["Score"],
                    "request_text": span,
                }
            )

    if not records:
        empty = pd.DataFrame(columns=["cluster", "request_text", "count", "avg_score"])
        empty.to_csv(settings.reports / "feature_requests.csv", index=False)
        return empty

    requests_df = pd.DataFrame(records)
    texts = requests_df["request_text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    k = min(max_clusters, max(2, len(texts) // 10))
    if k < 2 or X.shape[0] < 3:
        requests_df["cluster"] = 0
    else:
        model = KMeans(n_clusters=k, random_state=settings.random_seed, n_init=10)
        requests_df["cluster"] = model.fit_predict(X)
        try:
            sil = silhouette_score(X, requests_df["cluster"])
            logger.info("Feature request clustering silhouette: %.3f", sil)
        except ValueError:
            pass

    summary = (
        requests_df.groupby("cluster")
        .agg(
            count=("request_text", "count"),
            avg_score=("Score", "mean"),
            sample_request=("request_text", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(settings.reports / "feature_requests.csv", index=False)
    logger.info("Feature requests mined: %d hits, %d clusters", len(requests_df), summary.shape[0])
    return requests_df
