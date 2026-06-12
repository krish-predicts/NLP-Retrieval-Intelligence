"""Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        return None
    return px.bar(df, x=x, y=y, title=title)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        return None
    return px.line(df, x=x, y=y, title=title)


def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str | None, title: str):
    if df.empty:
        return None
    return px.scatter(df, x=x, y=y, color=color, title=title)
