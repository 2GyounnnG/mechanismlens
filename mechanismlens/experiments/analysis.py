"""Analysis helpers for synthetic MechanismLens experiments."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

from mechanismlens.benchmark import BenchmarkCase, BenchmarkResult
from mechanismlens.schema import AuditReport, CATEGORY_ORDER

SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 3.0}


def severity_weighted_risk_score(report_or_result: AuditReport | BenchmarkResult) -> float:
    """Compute a compact weighted score from severity counts."""

    counts = report_or_result.severity_counts
    if callable(counts):
        counts = counts()
    return float(sum(SEVERITY_WEIGHTS.get(severity, 0.0) * count for severity, count in counts.items()))


def extract_case_record(
    case: BenchmarkCase,
    report: AuditReport,
    result: BenchmarkResult,
    ground_truth_labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a flat-ish experiment record for JSON/CSV export."""

    injected_failures = list(ground_truth_labels.get("injected_failures", []))
    detected_categories = sorted(result.category_counts)
    decision_gap = result.metrics.get("decision_return_gap", {})
    return_gap = decision_gap.get("return_gap") if isinstance(decision_gap, dict) else None
    prediction_errors = result.metrics.get("mean_position_error", [])
    mean_prediction_error = (
        sum(float(value) for value in prediction_errors) / len(prediction_errors)
        if isinstance(prediction_errors, list) and prediction_errors
        else 0.0
    )
    return {
        "case_name": case.name,
        "description": case.description,
        "injected_failures": injected_failures,
        "detected_categories": detected_categories,
        "severity": ground_truth_labels.get("severity", "unknown"),
        "has_downstream_failure": bool(ground_truth_labels.get("has_downstream_failure", False)),
        "seed": ground_truth_labels.get("seed"),
        "overall_risk": result.overall_risk,
        "finding_count": result.finding_count,
        "risk_score": severity_weighted_risk_score(result),
        "category_counts": result.category_counts,
        "severity_counts": result.severity_counts,
        "metrics": result.metrics,
        "mean_prediction_error": mean_prediction_error,
        "return_gap": return_gap,
        "finding_messages": [finding.message for finding in report.findings],
    }


def compute_detection_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize detection rates by injected failure category."""

    categories = [*CATEGORY_ORDER, "clean", "combined"]
    by_category: dict[str, dict[str, int | float]] = {
        category: {"total": 0, "detected": 0, "missed": 0, "detection_rate": 0.0}
        for category in categories
    }

    for record in records:
        injected = record.get("injected_failures", [])
        detected = set(record.get("detected_categories", []))
        labels = ["clean"] if not injected else list(injected)
        if len(injected) > 1:
            labels.append("combined")
        for label in labels:
            if label not in by_category:
                by_category[label] = {"total": 0, "detected": 0, "missed": 0, "detection_rate": 0.0}
            by_category[label]["total"] += 1
            is_detected = not injected and not detected or label in detected
            if label == "combined":
                is_detected = all(item in detected for item in injected)
            if is_detected:
                by_category[label]["detected"] += 1
            else:
                by_category[label]["missed"] += 1

    for stats in by_category.values():
        total = int(stats["total"])
        stats["detection_rate"] = 0.0 if total == 0 else int(stats["detected"]) / total

    return {
        "num_cases": len(records),
        "by_injected_failure": by_category,
    }


def compute_category_confusion(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count injected category vs detected category pairs."""

    labels = [*CATEGORY_ORDER, "clean"]
    confusion = {label: {category: 0 for category in labels} for label in labels}
    for record in records:
        injected = record.get("injected_failures", []) or ["clean"]
        detected = record.get("detected_categories", []) or ["clean"]
        for injected_label in injected:
            confusion.setdefault(injected_label, {category: 0 for category in labels})
            for detected_label in detected:
                if detected_label not in confusion[injected_label]:
                    confusion[injected_label][detected_label] = 0
                confusion[injected_label][detected_label] += 1
    return confusion


def compute_risk_failure_correlation(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute simple Pearson correlations with risk score."""

    risk_scores = [float(record.get("risk_score", 0.0)) for record in records]
    downstream = [1.0 if record.get("has_downstream_failure", False) else 0.0 for record in records]
    return_gaps = [
        float(record["return_gap"])
        for record in records
        if isinstance(record.get("return_gap"), (int, float))
    ]
    return_gap_risks = [
        float(record.get("risk_score", 0.0))
        for record in records
        if isinstance(record.get("return_gap"), (int, float))
    ]
    return {
        "risk_downstream_failure_correlation": _pearson(risk_scores, downstream),
        "risk_return_gap_correlation": _pearson(return_gap_risks, return_gaps),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denom = sqrt(sum(value * value for value in left_centered)) * sqrt(
        sum(value * value for value in right_centered)
    )
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denom
