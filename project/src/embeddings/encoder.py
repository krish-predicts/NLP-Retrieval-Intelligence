"""SentenceTransformer encoder with disk caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.utils.exceptions import EmbeddingCacheError

logger = setup_logging(__name__)


class EmbeddingEncoder:
    """Encode text with optional per-review disk cache."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.default_embedding_model
        self.batch_size = settings.embedding_batch_size
        self.cache_dir = settings.faiss_dir / "cache" / self.model_name.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self, review_id: int | str) -> Path:
        return self.cache_dir / f"{review_id}.npy"

    def encode_single(self, text: str, review_id: int | str | None = None) -> np.ndarray:
        """Encode one text; use cache when review_id provided."""
        if review_id is not None:
            cache_path = self._cache_path(review_id)
            if cache_path.exists():
                return np.load(cache_path)

        vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        if review_id is not None:
            np.save(self._cache_path(review_id), vector)
        return vector

    def encode_batch(
        self,
        texts: list[str],
        review_ids: list[int | str] | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode batch with cache lookup for known IDs."""
        if review_ids is None:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )

        if len(texts) != len(review_ids):
            raise EmbeddingCacheError("texts and review_ids length mismatch")

        vectors: list[np.ndarray | None] = [None] * len(texts)
        pending_texts: list[str] = []
        pending_indices: list[int] = []

        for i, (text, rid) in enumerate(zip(texts, review_ids)):
            cache_path = self._cache_path(rid)
            if use_cache and cache_path.exists():
                vectors[i] = np.load(cache_path)
            else:
                pending_texts.append(text)
                pending_indices.append(i)

        if pending_texts:
            encoded = self.model.encode(
                pending_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(pending_texts) > 100,
            )
            for idx, vector in zip(pending_indices, encoded):
                vectors[idx] = vector
                if use_cache:
                    np.save(self._cache_path(review_ids[idx]), vector)

        return np.vstack([v for v in vectors if v is not None])
