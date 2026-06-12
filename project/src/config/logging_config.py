"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.settings import get_settings


def setup_logging(name: str = "nlp_retrieval", level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging."""
    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)
    log_path = settings.reports / "pipeline.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
