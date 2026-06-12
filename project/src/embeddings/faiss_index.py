"""FAISS index build, persist, and search."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.embeddings.encoder import EmbeddingEncoder
from src.utils.exceptions import IndexNotBuiltError
from src.utils.io import save_parquet

logger = setup_logging(__name__)


class FaissIndex:
    """FAISS IndexFlatIP with review id mapping and metadata."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.default_embedding_model
        self.index_dir = settings.faiss_dir / self.model_name.replace("/", "_")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = EmbeddingEncoder(self.model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: pd.DataFrame | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def id_map_path(self) -> Path:
        return self.index_dir / "id_map.parquet"

    def build(self, df: pd.DataFrame, text_col: str = "combined_text") -> None:
        """Build FAISS index from corpus."""
        texts = df[text_col].astype(str).tolist()
        review_ids = df["Id"].tolist()
        logger.info("Encoding %d documents for FAISS", len(texts))
        vectors = self.encoder.encode_batch(texts, review_ids=review_ids)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))

        self.index = index
        self.id_map = df.reset_index(drop=True).copy()
        self.id_map["faiss_idx"] = range(len(df))
        self.save()
        logger.info("FAISS index built: %d vectors, dim=%d", len(df), dim)

    def save(self) -> None:
        """Persist index and id map."""
        if self.index is None or self.id_map is None:
            raise IndexNotBuiltError("Cannot save: index not built")
        faiss.write_index(self.index, str(self.index_path))
        save_parquet(self.id_map, self.id_map_path)

    def load(self) -> None:
        """Load persisted index."""
        if not self.index_path.exists() or not self.id_map_path.exists():
            raise IndexNotBuiltError(f"Index not found in {self.index_dir}")
        self.index = faiss.read_index(str(self.index_path))
        self.id_map = pd.read_parquet(self.id_map_path)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[int, float, pd.Series]]:
        """Search index; optional metadata pre-filter."""
        if self.index is None or self.id_map is None:
            self.load()

        assert self.index is not None and self.id_map is not None
        candidate_map = self.id_map
        if filters:
            mask = pd.Series(True, index=candidate_map.index)
            for key, value in filters.items():
                if key in candidate_map.columns:
                    mask &= candidate_map[key] == value
            candidate_map = candidate_map[mask]
            if candidate_map.empty:
                return []

        query_vec = self.encoder.encode_single(query).astype(np.float32).reshape(1, -1)
        search_k = min(top_k * 5 if filters else top_k, len(candidate_map))
        scores, indices = self.index.search(query_vec, search_k)

        results: list[tuple[int, float, pd.Series]] = []
        allowed = set(candidate_map["faiss_idx"].tolist()) if filters else None
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if allowed is not None and idx not in allowed:
                continue
            row = self.id_map[self.id_map["faiss_idx"] == idx].iloc[0]
            results.append((int(row["Id"]), float(score), row))
            if len(results) >= top_k:
                break
        return results
