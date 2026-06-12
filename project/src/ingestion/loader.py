"""Load raw review CSV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.logging_config import setup_logging

logger = setup_logging(__name__)


def load_reviews(path: Path, chunksize: int | None = None) -> pd.DataFrame:
    """Load reviews CSV with optional chunking."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    logger.info("Loading reviews from %s", path)
    if chunksize:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path)

    logger.info("Loaded %d rows", len(df))
    return df
