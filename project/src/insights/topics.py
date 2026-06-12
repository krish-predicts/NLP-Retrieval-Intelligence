"""BERTopic topic modeling."""

from __future__ import annotations

import pandas as pd
from bertopic import BERTopic

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)


def run_topic_modeling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic and export topic summary."""
    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)

    fit_size = min(settings.topic_fit_sample, len(df))
    fit_df = df.sample(n=fit_size, random_state=settings.random_seed)
    docs = fit_df["combined_text"].astype(str).tolist()

    logger.info("Fitting BERTopic on %d documents", len(docs))
    model = BERTopic(min_topic_size=30, verbose=False)
    topics, _ = model.fit_transform(docs)

    fit_df = fit_df.copy()
    fit_df["topic_id"] = topics

    all_docs = df["combined_text"].astype(str).tolist()
    all_topics, _ = model.transform(all_docs)
    enriched = df.copy()
    enriched["topic_id"] = all_topics

    topic_info = model.get_topic_info()
    summary_rows: list[dict[str, object]] = []
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        topic_docs = enriched[enriched["topic_id"] == tid]
        top_terms = model.get_topic(tid)
        label = "_".join([term for term, _ in (top_terms or [])[:3]]) if top_terms else f"topic_{tid}"
        summary_rows.append(
            {
                "topic_id": tid,
                "label": label,
                "count": len(topic_docs),
                "avg_score": float(topic_docs["Score"].mean()) if len(topic_docs) else 0.0,
                "representative_terms": ", ".join([t for t, _ in (top_terms or [])[:8]]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
    summary.to_csv(settings.reports / "topic_summary.csv", index=False)
    logger.info("Topic modeling complete: %d topics", len(summary))
    return enriched, summary
