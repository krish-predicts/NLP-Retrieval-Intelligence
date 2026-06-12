"""Pydantic schemas for review records."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ReviewRecord(BaseModel):
    """Validated Amazon Fine Food review record."""

    Id: int
    ProductId: str
    UserId: str
    ProfileName: str | None = None
    HelpfulnessNumerator: int = 0
    HelpfulnessDenominator: int = 0
    Score: int = Field(ge=1, le=5)
    Time: int
    Summary: str | None = None
    Text: str

    @field_validator("Text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text must not be empty")
        return value

    @field_validator("Summary", "ProfileName", mode="before")
    @classmethod
    def coerce_none(cls, value: object) -> str | None:
        if value is None or (isinstance(value, float) and str(value) == "nan"):
            return None
        return str(value) if value else None
