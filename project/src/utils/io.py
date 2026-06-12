"""I/O utilities for parquet and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    """Create parent directories if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet."""
    ensure_parent(path)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet."""
    return pd.read_parquet(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dict as JSON."""
    ensure_parent(path)

    def _default(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_default)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
