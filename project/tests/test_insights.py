"""Insight mining tests."""

import pandas as pd

from src.insights.defects import mine_defects
from src.insights.feature_requests import mine_feature_requests
from src.preprocessing.cleaner import clean_reviews
from src.preprocessing.metadata import add_metadata


def _sample_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "ProductId": ["P1", "P2", "P3"],
            "Score": [1, 4, 2],
            "Time": [1303862400, 1346976000, 1219017600],
            "Summary": ["Bad", "Good", "Ok"],
            "Text": [
                "Product was broken and defective",
                "Great taste love it",
                "Wish it had better packaging and missing label",
            ],
            "HelpfulnessNumerator": [0, 1, 0],
            "HelpfulnessDenominator": [0, 1, 0],
        }
    )
    df = clean_reviews(df)
    return add_metadata(df)


def test_feature_request_mining():
    df = _sample_df()
    result = mine_feature_requests(df, max_clusters=2)
    assert not result.empty


def test_defect_mining():
    df = _sample_df()
    result = mine_defects(df)
    assert not result.empty
    assert "defect_type" in result.columns
