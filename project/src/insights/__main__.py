"""Run insights pipeline from CLI."""

from __future__ import annotations

from src.utils.io import load_parquet
from src.utils.multiprocessing_guard import configure_multiprocessing


def main() -> None:
    configure_multiprocessing()

    from src.config.settings import get_settings
    from src.insights.pipeline import run_insights

    settings = get_settings()
    df = load_parquet(settings.processed)
    run_insights(df)


if __name__ == "__main__":
    main()
