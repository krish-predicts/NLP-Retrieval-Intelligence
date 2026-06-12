"""Preprocessing tests."""

import pandas as pd

from src.preprocessing.cleaner import clean_text
from src.preprocessing.metadata import add_metadata


def test_clean_text_strips_html():
    text = "Hello<br />world"
    assert "<br" not in clean_text(text)


def test_add_metadata_fields():
    df = pd.DataFrame(
        {
            "Id": [1],
            "Summary": ["Good"],
            "Text": ["Nice product"],
            "Score": [5],
            "Time": [1303862400],
            "HelpfulnessNumerator": [1],
            "HelpfulnessDenominator": [1],
        }
    )
    enriched = add_metadata(df)
    assert "combined_text" in enriched.columns
    assert enriched.loc[0, "word_count"] >= 2
    assert "year_month" in enriched.columns
