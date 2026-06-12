"""Topic quality metrics."""

from __future__ import annotations

from typing import Any


def topic_coherence_proxy(topics: list[dict[str, Any]]) -> float:
    """Simple coherence proxy: average unique term overlap in topic labels."""
    if not topics:
        return 0.0
    scores: list[float] = []
    for topic in topics:
        terms = str(topic.get("label", "")).split("_")
        if len(terms) >= 2:
            scores.append(min(1.0, len(set(terms)) / len(terms)))
    return sum(scores) / len(scores) if scores else 0.0
