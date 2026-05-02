"""Batch benchmark utilities for MechanismLens toy audit suites."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mechanismlens.audit_suite import AuditSuite
from mechanismlens.schema import AuditInput, AuditReport, CATEGORY_ORDER


@dataclass
class BenchmarkCase:
    """One predefined audit case."""

    name: str
    description: str
    audit_input: AuditInput
    contract_name: str | None = None


@dataclass
class BenchmarkResult:
    """Compact benchmark result suitable for aggregate output."""

    case_name: str
    overall_risk: str
    finding_count: int
    category_counts: dict[str, int]
    severity_counts: dict[str, int]
    metrics: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "case_name": self.case_name,
            "overall_risk": self.overall_risk,
            "finding_count": self.finding_count,
            "category_counts": self.category_counts,
            "severity_counts": self.severity_counts,
            "metrics": self.metrics,
        }


class BenchmarkRunner:
    """Run multiple benchmark cases through an audit suite."""

    CSV_HEADERS = [
        "case_name",
        "overall_risk",
        "finding_count",
        "low_count",
        "medium_count",
        "high_count",
        "semantic_count",
        "causal_count",
        "physics_count",
        "cross_layer_count",
        "decision_count",
        "horizon_count",
    ]

    def __init__(self, audit_suite: AuditSuite | None = None) -> None:
        self.audit_suite = audit_suite or AuditSuite()

    def run_case(self, case: BenchmarkCase) -> tuple[AuditReport, BenchmarkResult]:
        """Run one benchmark case and return the full and compact reports."""

        report = self.audit_suite.run(case.audit_input)
        result = BenchmarkResult(
            case_name=case.name,
            overall_risk=report.overall_risk,
            finding_count=len(report.findings),
            category_counts=report.category_counts(),
            severity_counts=report.severity_counts(),
            metrics=report.metrics,
        )
        return report, result

    def run(self, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
        """Run benchmark cases in order."""

        return [self.run_case(case)[1] for case in cases]

    def save_summary_json(self, results: list[BenchmarkResult], path: str | Path) -> None:
        """Save benchmark summary as deterministic JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.to_json_dict() for result in results]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_summary_csv(self, results: list[BenchmarkResult], path: str | Path) -> None:
        """Save benchmark summary as CSV."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            for result in results:
                writer.writerow(self._csv_row(result))

    def _csv_row(self, result: BenchmarkResult) -> dict[str, Any]:
        return {
            "case_name": result.case_name,
            "overall_risk": result.overall_risk,
            "finding_count": result.finding_count,
            "low_count": result.severity_counts.get("low", 0),
            "medium_count": result.severity_counts.get("medium", 0),
            "high_count": result.severity_counts.get("high", 0),
            **{
                f"{category}_count": result.category_counts.get(category, 0)
                for category in CATEGORY_ORDER
            },
        }
