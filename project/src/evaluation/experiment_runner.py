"""Retrieval benchmark and experiment runner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.evaluation.pseudo_qrels import build_pseudo_qrels
from src.evaluation.queries import EVAL_QUERIES
from src.evaluation.retrieval_metrics import (
    latency_stats,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.registry import RetrieverName, get_retriever

logger = setup_logging(__name__)


def evaluate_retriever(
    corpus: pd.DataFrame,
    retriever_name: RetrieverName,
    queries: dict[str, list[str]] | None = None,
    top_k: int | None = None,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate retriever and return benchmark + retrieved results."""
    settings = get_settings()
    k = top_k or settings.top_k
    query_map = queries or EVAL_QUERIES

    retriever = get_retriever(retriever_name, model_name=model_name)
    retriever.fit(corpus)

    benchmark_rows: list[dict[str, Any]] = []
    retrieved_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for query, terms in query_map.items():
        relevant = build_pseudo_qrels(corpus, query, terms)
        results = retriever.search(query, top_k=k)
        retrieved_ids = [r.review_id for r in results]
        if results:
            latencies.append(results[0].latency_ms)

        benchmark_rows.append(
            {
                "query": query,
                "retriever": retriever_name,
                "model": model_name or settings.default_embedding_model,
                f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
                f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
                "mrr": mrr(retrieved_ids, relevant),
                f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant, k),
                "latency_ms": results[0].latency_ms if results else 0.0,
                "relevant_count": len(relevant),
            }
        )

        for rank, result in enumerate(results, start=1):
            retrieved_rows.append(
                {
                    "query": query,
                    "retriever": retriever_name,
                    "rank": rank,
                    "review_id": result.review_id,
                    "score": result.score,
                    "latency_ms": result.latency_ms,
                    "is_relevant": result.review_id in relevant,
                    "text": result.text[:500],
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    latency_summary = latency_stats(latencies)
    logger.info("Retriever %s latency: %s", retriever_name, latency_summary)

    retrieved_df = pd.DataFrame(retrieved_rows)
    return benchmark_df, retrieved_df


def run_full_benchmark(corpus: pd.DataFrame) -> None:
    """Run all retrievers and save benchmark outputs."""
    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)

    all_benchmark: list[pd.DataFrame] = []
    all_retrieved: list[pd.DataFrame] = []

    for name in ["tfidf", "bm25", "dense", "hybrid"]:
        bench, retrieved = evaluate_retriever(corpus, name)  # type: ignore[arg-type]
        all_benchmark.append(bench)
        all_retrieved.append(retrieved)

    benchmark = pd.concat(all_benchmark, ignore_index=True)
    retrieved = pd.concat(all_retrieved, ignore_index=True)

    benchmark.to_csv(settings.reports / "retrieval_benchmark.csv", index=False)
    retrieved.to_csv(settings.reports / "retrieved_results.csv", index=False)
    logger.info("Saved retrieval benchmark and retrieved results")
