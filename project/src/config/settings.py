"""Application configuration via Pydantic settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Config-driven paths and hyperparameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_default_project_root)

    raw_data_path: str = "data/raw/Reviews.csv"
    processed_raw_path: str = "data/processed/reviews_raw.parquet"
    processed_path: str = "data/processed/reviews.parquet"
    faiss_index_dir: str = "data/embeddings"
    reports_dir: str = "reports"

    sample_size: int = 50000
    random_seed: int = 42

    default_embedding_model: str = "all-mpnet-base-v2"
    embedding_models: list[str] = Field(
        default_factory=lambda: ["all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
    )

    hybrid_alpha: float = 0.5
    top_k: int = 10

    chunk_max_tokens: int = 128
    chunk_size: int = 64
    chunk_overlap: int = 16
    chunk_strategy: Literal["fixed", "overlap", "semantic"] = "fixed"

    lowercase_text: bool = False
    topic_fit_sample: int = 20000
    sentiment_train_sample: int = 10000
    embedding_batch_size: int = 64

    @field_validator("embedding_models", mode="before")
    @classmethod
    def parse_embedding_models(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value  # type: ignore[return-value]

    def resolve(self, relative: str) -> Path:
        """Resolve a path relative to project root."""
        path = Path(relative)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def raw_data(self) -> Path:
        return self.resolve(self.raw_data_path)

    @property
    def processed_raw(self) -> Path:
        return self.resolve(self.processed_raw_path)

    @property
    def processed(self) -> Path:
        return self.resolve(self.processed_path)

    @property
    def faiss_dir(self) -> Path:
        return self.resolve(self.faiss_index_dir)

    @property
    def reports(self) -> Path:
        return self.resolve(self.reports_dir)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
