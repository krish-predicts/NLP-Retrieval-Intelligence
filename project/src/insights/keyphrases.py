"""KeyBERT keyphrase extraction."""

from __future__ import annotations

import pandas as pd
from keybert import KeyBERT

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)


def extract_keyphrases(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract top complaint and positive keyphrases."""
    settings = get_settings()
    kw_model = KeyBERT()

    low = df[df["Score"] <= 2]["combined_text"].astype(str).tolist()[:500]
    high = df[df["Score"] >= 4]["combined_text"].astype(str).tolist()[:500]

    rows: list[dict[str, object]] = []
    if low:
        sample_text = " ".join(low[:100])
        for phrase, score in kw_model.extract_keywords(sample_text, top_n=top_n):
            rows.append({"category": "complaint", "phrase": phrase, "score": float(score)})
    if high:
        sample_text = " ".join(high[:100])
        for phrase, score in kw_model.extract_keywords(sample_text, top_n=top_n):
            rows.append({"category": "positive", "phrase": phrase, "score": float(score)})

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(settings.reports / "keyphrases.csv", index=False)
    logger.info("Extracted %d keyphrases", len(result))
    return result
