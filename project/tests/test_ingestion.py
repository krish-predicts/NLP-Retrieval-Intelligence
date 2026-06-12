"""Ingestion tests."""

from pathlib import Path

import pandas as pd

from src.ingestion.deduplicator import deduplicate_reviews
from src.ingestion.loader import load_reviews
from src.ingestion.sampler import stratified_sample
from src.ingestion.validator import validate_reviews


FIXTURE = Path(__file__).parent / "fixtures" / "sample_reviews.csv"


def test_load_reviews():
    df = load_reviews(FIXTURE)
    assert len(df) == 10


def test_validate_reviews():
    df = load_reviews(FIXTURE)
    validated, report = validate_reviews(df)
    assert report["valid_rows"] == 10
    assert len(validated) == 10


def test_deduplicate_reviews():
    df = load_reviews(FIXTURE)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    deduped = deduplicate_reviews(dup)
    assert len(deduped) == 10


def test_stratified_sample():
    df = load_reviews(FIXTURE)
    sampled = stratified_sample(df, sample_size=5)
    assert len(sampled) == 5
