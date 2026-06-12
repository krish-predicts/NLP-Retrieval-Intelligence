"""Timing utilities for latency measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[list[float], None, None]:
    """Context manager that yields a list with elapsed ms appended on exit."""
    result: list[float] = []
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.append((time.perf_counter() - start) * 1000.0)
