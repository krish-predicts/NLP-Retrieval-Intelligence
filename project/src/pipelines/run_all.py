"""Main pipeline orchestrator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.embeddings.faiss_index import FaissIndex
from src.evaluation.experiment_runner import run_full_benchmark
from src.ingestion.pipeline import run_ingestion
from src.insights.pipeline import run_insights, run_sentiment_stage
from src.preprocessing.pipeline import run_preprocessing
from src.utils.io import load_parquet, save_json
from src.utils.multiprocessing_guard import configure_multiprocessing

logger = setup_logging(__name__)

STAGES = [
    "ingestion",
    "preprocessing",
    "embeddings",
    "insights",
    "sentiment",
    "evaluation",
    "all",
]


def run_embeddings() -> None:
    settings = get_settings()
    df = load_parquet(settings.processed)
    for model in settings.embedding_models:
        index = FaissIndex(model)
        if not index.index_path.exists():
            index.build(df)
        else:
            logger.info("FAISS index exists for %s, skipping build", model)


def write_manifest(row_count: int) -> None:
    settings = get_settings()
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "sample_size": settings.sample_size,
        "random_seed": settings.random_seed,
        "embedding_models": settings.embedding_models,
    }
    save_json(manifest, settings.reports / "run_manifest.json")


def main() -> None:
    configure_multiprocessing()

    parser = argparse.ArgumentParser(description="NLP Retrieval Intelligence Pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    args = parser.parse_args()

    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)

    df = None
    if args.stage in ("ingestion", "all"):
        df = run_ingestion()
    if args.stage in ("preprocessing", "all"):
        df = run_preprocessing(df)
    if args.stage in ("embeddings", "all"):
        if df is None:
            df = load_parquet(settings.processed)
        run_embeddings()
    if args.stage in ("insights", "all"):
        if df is None:
            df = load_parquet(settings.processed)
        df = run_insights(df)
    if args.stage in ("sentiment",):
        if df is None:
            df = load_parquet(settings.processed)
        df = run_sentiment_stage(df)
    if args.stage in ("evaluation", "all"):
        if df is None:
            df = load_parquet(settings.processed)
        run_full_benchmark(df)

    if df is None:
        df = load_parquet(settings.processed) if settings.processed.exists() else None
    if df is not None:
        write_manifest(len(df))
    logger.info("Pipeline stage '%s' complete", args.stage)


if __name__ == "__main__":
    main()
