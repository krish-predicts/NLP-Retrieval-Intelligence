"""Stakeholder report generation and PDF export."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fpdf import FPDF

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.utils.io import save_json

logger = setup_logging(__name__)


def _safe_text(text: str, max_len: int = 200) -> str:
    cleaned = str(text).encode("latin-1", errors="replace").decode("latin-1")
    return cleaned[:max_len]


def build_stakeholder_summary(
    df: pd.DataFrame,
    topic_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    defect_summary: pd.DataFrame,
    keyphrases: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build stakeholder-facing summary dict."""
    rating_trend = df.groupby("year_month")["Score"].mean().tail(6).to_dict()
    low_topics = topic_summary.nsmallest(5, "avg_score")[["label", "count", "avg_score"]].to_dict(
        orient="records"
    )

    escalation = []
    if not defect_summary.empty and "immediate_action" in defect_summary.columns:
        escalation = defect_summary[defect_summary["immediate_action"] == True].head(5).to_dict(  # noqa: E712
            orient="records"
        )

    return {
        "executive": {
            "total_reviews": len(df),
            "avg_rating": float(df["Score"].mean()),
            "rating_trend_6m": rating_trend,
            "emerging_risks": defect_summary.head(5).to_dict(orient="records"),
            "opportunities": feature_summary.head(5).to_dict(orient="records"),
        },
        "product": {
            "top_requested_features": feature_summary.head(10).to_dict(orient="records"),
            "top_defects": defect_summary.head(10).to_dict(orient="records"),
            "recommendations": [
                "Address top defect themes in QA checklist",
                "Prioritize features with highest request volume",
                "Monitor products with declining avg_score",
            ],
        },
        "customer_experience": {
            "top_complaint_topics": low_topics,
            "escalation_priorities": escalation,
        },
        "marketing": {
            "positive_themes": topic_summary.nlargest(5, "avg_score")[["label", "count"]].to_dict(
                orient="records"
            ),
            "purchase_drivers": (
                keyphrases[keyphrases["category"] == "positive"].head(10).to_dict(orient="records")
                if keyphrases is not None and not keyphrases.empty
                else []
            ),
        },
        "business_questions": {
            "why_ratings_change": rating_trend,
            "common_complaints": low_topics,
            "feature_requests": feature_summary.head(10).to_dict(orient="records"),
            "positive_drivers": topic_summary.nlargest(5, "avg_score").to_dict(orient="records"),
            "emerging_defects": defect_summary.head(10).to_dict(orient="records"),
            "immediate_action": escalation,
            "low_rating_topics": low_topics,
        },
    }


def _export_stakeholder_pdf(summary: dict[str, Any], output_path: str) -> None:
    """Write stakeholder summary PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NLP Retrieval Intelligence - Stakeholder Summary", ln=True)
    pdf.set_font("Helvetica", size=11)

    exec_summary = summary.get("executive", {})
    pdf.cell(0, 8, _safe_text(f"Total Reviews: {exec_summary.get('total_reviews', 0)}"), ln=True)
    pdf.cell(0, 8, _safe_text(f"Avg Rating: {exec_summary.get('avg_rating', 0):.2f}"), ln=True)

    for section, title in [
        ("product", "Product Team"),
        ("customer_experience", "Customer Experience"),
        ("marketing", "Marketing"),
        ("executive", "Leadership"),
    ]:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", size=10)
        content = summary.get(section, {})
        for key, value in list(content.items())[:4]:
            pdf.multi_cell(0, 6, _safe_text(f"- {key}: {value}"))

    pdf.output(output_path)


def generate_stakeholder_report(
    df: pd.DataFrame,
    topic_summary: pd.DataFrame,
    feature_df: pd.DataFrame,
    defect_df: pd.DataFrame,
    keyphrases: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate JSON + PDF stakeholder outputs from insight artifacts."""
    settings = get_settings()
    settings.reports.mkdir(parents=True, exist_ok=True)

    feature_path = settings.reports / "feature_requests.csv"
    defect_path = settings.reports / "defect_report.csv"
    feature_summary = (
        pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()
    )
    defect_summary = pd.read_csv(defect_path) if defect_path.exists() else pd.DataFrame()

    summary = build_stakeholder_summary(
        df, topic_summary, feature_summary, defect_summary, keyphrases
    )
    save_json(summary, settings.reports / "stakeholder_summary.json")

    pdf_path = str(settings.reports / "stakeholder_summary.pdf")
    _export_stakeholder_pdf(summary, pdf_path)
    logger.info("Stakeholder report saved to %s", pdf_path)
    return summary
