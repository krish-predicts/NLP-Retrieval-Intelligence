"""Sentiment analysis with rating alignment."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)


def _rating_to_label(score: int) -> str | None:
    if score >= 4:
        return "positive"
    if score <= 2:
        return "negative"
    return None


def run_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Train TF-IDF + LogisticRegression sentiment model; predict all reviews."""
    settings = get_settings()
    labeled = df.copy()
    labeled["sentiment_label"] = labeled["Score"].map(_rating_to_label)
    train_df = labeled[labeled["sentiment_label"].notna()].sample(
        n=min(settings.sentiment_train_sample, len(labeled)),
        random_state=settings.random_seed,
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_df["combined_text"].astype(str))
    y = train_df["sentiment_label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=settings.random_seed)
    clf.fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info("Sentiment classifier test accuracy: %.3f", test_acc)

    all_X = vectorizer.transform(df["combined_text"].astype(str))
    predictions = clf.predict(all_X)
    result = df.copy()
    result["sentiment"] = predictions

    def aligned(row: pd.Series) -> bool:
        label = _rating_to_label(int(row["Score"]))
        if label is None:
            return True
        return row["sentiment"] == label

    result["sentiment_aligned"] = result.apply(aligned, axis=1)
    alignment_rate = result["sentiment_aligned"].mean()
    logger.info("Rating-sentiment alignment: %.2f%%", alignment_rate * 100)
    return result
