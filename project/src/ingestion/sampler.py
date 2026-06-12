"""Stratified sampling by rating score."""

from __future__ import annotations

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings

logger = setup_logging(__name__)


def stratified_sample(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """Return stratified sample by Score; sample_size=0 returns full dataset."""
    settings = get_settings()
    n = sample_size if sample_size is not None else settings.sample_size

    if n <= 0 or n >= len(df):
        logger.info("Using full dataset (%d rows)", len(df))
        return df.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("Score"):
        group_n = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(n=min(group_n, len(group)), random_state=settings.random_seed))

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=settings.random_seed).reset_index(drop=True)

    logger.info("Stratified sample: %d rows (target %d)", len(sampled), n)
    return sampled
