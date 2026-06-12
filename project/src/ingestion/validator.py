"""Schema validation and missing-value handling."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.config.logging_config import setup_logging
from src.ingestion.schemas import ReviewRecord
from src.utils.exceptions import SchemaValidationError

logger = setup_logging(__name__)


def validate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate rows against ReviewRecord schema; drop invalid rows."""
    valid_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            record = ReviewRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except ValidationError as exc:
            errors.append(f"Row {idx}: {exc}")

    if not valid_rows:
        raise SchemaValidationError("No valid rows after schema validation")

    validated = pd.DataFrame(valid_rows)
    missing_report = {
        "total_input": len(df),
        "valid_rows": len(validated),
        "dropped_rows": len(df) - len(validated),
        "validation_errors_sample": errors[:20],
        "missing_values": validated.isnull().sum().to_dict(),
    }

    logger.info(
        "Validation complete: %d valid / %d input",
        missing_report["valid_rows"],
        missing_report["total_input"],
    )
    return validated, missing_report
