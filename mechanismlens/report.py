"""Report helpers for MechanismLens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .recommendations import generate_recommendations
from .schema import AuditReport


def to_json_dict(report: AuditReport) -> dict[str, Any]:
    """Return a JSON-friendly report dictionary."""

    return report.to_json_dict()


def save_json(report: AuditReport, path: str | Path) -> None:
    """Save a report as JSON."""

    report.save_json(path)


def save_markdown(report: AuditReport, path: str | Path) -> None:
    """Save a report as Markdown."""

    report.save_markdown(path)


def category_counts(report: AuditReport) -> dict[str, int]:
    """Count findings by category."""

    return report.category_counts()


def severity_counts(report: AuditReport) -> dict[str, int]:
    """Count findings by severity."""

    return report.severity_counts()


def risk_summary_table_markdown(report: AuditReport) -> str:
    """Render the report risk summary table."""

    return report.risk_summary_table_markdown()


__all__ = [
    "AuditReport",
    "category_counts",
    "generate_recommendations",
    "risk_summary_table_markdown",
    "save_json",
    "save_markdown",
    "severity_counts",
    "to_json_dict",
]
