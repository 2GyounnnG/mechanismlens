"""Analysis helpers for controlled synthetic MechanismLens experiments."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from statistics import median
from typing import Any

from mechanismlens.benchmark import BenchmarkCase, BenchmarkResult
from mechanismlens.schema import AuditReport, CATEGORY_ORDER, SEVERITY_ORDER

SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 4.0}
RECALL_CATEGORIES = ("physics", "causal", "cross_layer", "decision")


def severity_weighted_risk_score_from_counts(severity_counts: Mapping[str, int]) -> float:
    """Compute a compact weighted risk score from severity counts."""

    return float(
        sum(
            SEVERITY_WEIGHTS.get(severity, 0.0) * int(count)
            for severity, count in severity_counts.items()
        )
    )


def severity_weighted_risk_score(report_or_result: AuditReport | BenchmarkResult) -> float:
    """Compute a weighted score from an audit report or benchmark result."""

    counts = report_or_result.severity_counts
    if callable(counts):
        counts = counts()
    return severity_weighted_risk_score_from_counts(counts)


def extract_case_record(
    case: BenchmarkCase,
    report: AuditReport,
    result: BenchmarkResult,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one flat experiment record for JSON/CSV export."""

    category_counts = {
        category: result.category_counts.get(category, 0) for category in CATEGORY_ORDER
    }
    severity_counts = {
        severity: result.severity_counts.get(severity, 0) for severity in SEVERITY_ORDER
    }
    mean_position_error = result.metrics.get("mean_position_error", [])
    if isinstance(mean_position_error, list) and mean_position_error:
        numeric_errors = [
            float(value) for value in mean_position_error if isinstance(value, (int, float))
        ]
    else:
        numeric_errors = []
    decision_gap = result.metrics.get("decision_return_gap", {})
    return_gap = decision_gap.get("return_gap") if isinstance(decision_gap, Mapping) else None

    return {
        "case_id": labels.get("case_id", case.name),
        "case_name": case.name,
        "failure_type": labels.get("failure_type", "unknown"),
        "injected_failures": list(labels.get("injected_failures", [])),
        "expected_risk": labels.get("expected_risk", "unknown"),
        "overall_risk": result.overall_risk,
        "finding_count": result.finding_count,
        "low_count": severity_counts["low"],
        "medium_count": severity_counts["medium"],
        "high_count": severity_counts["high"],
        "semantic_count": category_counts["semantic"],
        "causal_count": category_counts["causal"],
        "physics_count": category_counts["physics"],
        "cross_layer_count": category_counts["cross_layer"],
        "decision_count": category_counts["decision"],
        "horizon_count": category_counts["horizon"],
        "risk_score": severity_weighted_risk_score(result),
        "mean_position_error_final": numeric_errors[-1] if numeric_errors else None,
        "mean_position_error_mean": (
            sum(numeric_errors) / len(numeric_errors) if numeric_errors else None
        ),
        "return_gap": float(return_gap) if isinstance(return_gap, (int, float)) else None,
        "has_downstream_failure": bool(labels.get("has_downstream_failure", False)),
        "expected_low_mse": bool(labels.get("expected_low_mse", False)),
        "severity": labels.get("severity", "unknown"),
        "seed": labels.get("seed"),
        "finding_messages": [finding.message for finding in report.findings],
    }


def compute_detection_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize case counts and detected categories by failure type."""

    by_failure_type: dict[str, dict[str, Any]] = {}
    detected_category_counts = {category: 0 for category in CATEGORY_ORDER}

    for record in records:
        failure_type = str(record.get("failure_type", "unknown"))
        stats = by_failure_type.setdefault(
            failure_type,
            {
                "total": 0,
                "detected": 0,
                "missed": 0,
                "detected_category_counts": {category: 0 for category in CATEGORY_ORDER},
            },
        )
        stats["total"] += 1
        detected_any = int(record.get("finding_count", 0)) > 0
        if detected_any:
            stats["detected"] += 1
        else:
            stats["missed"] += 1

        for category in CATEGORY_ORDER:
            count = int(record.get(f"{category}_count", 0))
            if count > 0:
                detected_category_counts[category] += 1
                stats["detected_category_counts"][category] += 1

    for stats in by_failure_type.values():
        total = int(stats["total"])
        stats["detection_rate"] = 0.0 if total == 0 else int(stats["detected"]) / total

    return {
        "num_cases": len(records),
        "by_failure_type": by_failure_type,
        "detected_category_counts": detected_category_counts,
    }


def compute_category_recall(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Estimate recall for injected mechanism categories."""

    recall: dict[str, dict[str, float | int]] = {}
    for category in RECALL_CATEGORIES:
        total = 0
        detected = 0
        for record in records:
            injected = set(record.get("injected_failures", []))
            if category not in injected:
                continue
            total += 1
            detected += int(int(record.get(f"{category}_count", 0)) > 0)
        recall[category] = {
            "total": total,
            "detected": detected,
            "recall": 0.0 if total == 0 else detected / total,
        }
    return recall


def compute_risk_by_failure_type(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Compute mean and median risk score by synthetic failure type."""

    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record.get("failure_type", "unknown")), []).append(
            float(record.get("risk_score", 0.0))
        )
    return {
        failure_type: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": float(median(values)),
        }
        for failure_type, values in sorted(grouped.items())
        if values
    }


def compute_metric_correlations(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute simple Pearson correlations without requiring pandas."""

    return {
        "risk_score_vs_has_downstream_failure": _correlate_present(
            records, "risk_score", "has_downstream_failure"
        ),
        "risk_score_vs_return_gap": _correlate_present(records, "risk_score", "return_gap"),
        "mean_position_error_mean_vs_return_gap": _correlate_present(
            records, "mean_position_error_mean", "return_gap"
        ),
        "mean_position_error_mean_vs_has_downstream_failure": _correlate_present(
            records, "mean_position_error_mean", "has_downstream_failure"
        ),
    }


def summarize_low_mse_high_risk(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count expected low-MSE cases that still receive a high mechanism-risk score."""

    low_mse_records = [record for record in records if bool(record.get("expected_low_mse", False))]
    high_risk_records = [
        record
        for record in low_mse_records
        if float(record.get("risk_score", 0.0)) >= SEVERITY_WEIGHTS["high"]
        or record.get("overall_risk") == "high"
    ]
    return {
        "expected_low_mse_count": len(low_mse_records),
        "low_mse_high_risk_count": len(high_risk_records),
    }


def _correlate_present(
    records: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> float | None:
    left: list[float] = []
    right: list[float] = []
    for record in records:
        left_value = _numeric_value(record.get(left_key))
        right_value = _numeric_value(record.get(right_key))
        if left_value is None or right_value is None:
            continue
        left.append(left_value)
        right.append(right_value)
    return _pearson(left, right)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


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
