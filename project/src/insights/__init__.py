"""Insights package — lazy exports to avoid import cycles."""

from __future__ import annotations

from typing import Any

__all__ = ["run_insights"]


def __getattr__(name: str) -> Any:
    if name == "run_insights":
        from src.insights.pipeline import run_insights

        return run_insights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
