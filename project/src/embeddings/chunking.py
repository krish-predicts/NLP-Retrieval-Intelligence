"""Text chunking strategies for retrieval experiments."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from src.config.settings import get_settings

ChunkStrategy = Literal["fixed", "overlap", "semantic"]


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def _chunk_words(words: list[str], size: int, overlap: int = 0) -> list[str]:
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = words[start : start + size]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + size >= len(words):
            break
    return chunks


def _semantic_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Split text using fixed, overlapping, or semantic chunking."""
    settings = get_settings()
    strategy = strategy or settings.chunk_strategy
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    max_tokens = max_tokens or settings.chunk_max_tokens

    words = _tokenize_words(text)
    if len(words) <= max_tokens:
        return [text]

    if strategy == "semantic":
        sentences = _semantic_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sent_words = _tokenize_words(sentence)
            if current_len + len(sent_words) > chunk_size and current:
                chunks.append(" ".join(current))
                current = sent_words
                current_len = len(sent_words)
            else:
                current.extend(sent_words)
                current_len += len(sent_words)
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    if strategy == "overlap":
        return _chunk_words(words, chunk_size, chunk_overlap)

    return _chunk_words(words, chunk_size, overlap=0)


def build_chunk_corpus(
    df: pd.DataFrame,
    text_col: str = "combined_text",
    id_col: str = "Id",
    strategy: ChunkStrategy | None = None,
) -> pd.DataFrame:
    """Expand reviews into chunks with parent review mapping."""
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        chunks = chunk_text(str(row[text_col]), strategy=strategy)
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{row[id_col]}_{idx}",
                    "review_id": row[id_col],
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "Score": row.get("Score"),
                    "ProductId": row.get("ProductId"),
                    "year_month": row.get("year_month"),
                }
            )
    return pd.DataFrame(records)
