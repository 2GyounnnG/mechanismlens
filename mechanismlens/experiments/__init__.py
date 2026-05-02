"""Synthetic experiment utilities for MechanismLens."""

from .analysis import (
    compute_category_recall,
    compute_detection_summary,
    compute_metric_correlations,
    compute_risk_by_failure_type,
    extract_case_record,
    severity_weighted_risk_score,
    severity_weighted_risk_score_from_counts,
    summarize_low_mse_high_risk,
)
from .synthetic_generator import (
    SyntheticCaseSpec,
    generate_causal_side_effect_case,
    generate_clean_case,
    generate_combined_failure_case,
    generate_cross_layer_failure_case,
    generate_decision_risk_case,
    generate_low_mse_high_risk_case,
    generate_physics_failure_case,
    generate_prediction_error_case,
    generate_synthetic_suite,
)

__all__ = [
    "SyntheticCaseSpec",
    "compute_category_recall",
    "compute_detection_summary",
    "compute_metric_correlations",
    "compute_risk_by_failure_type",
    "extract_case_record",
    "generate_causal_side_effect_case",
    "generate_clean_case",
    "generate_combined_failure_case",
    "generate_cross_layer_failure_case",
    "generate_decision_risk_case",
    "generate_low_mse_high_risk_case",
    "generate_physics_failure_case",
    "generate_prediction_error_case",
    "generate_synthetic_suite",
    "severity_weighted_risk_score",
    "severity_weighted_risk_score_from_counts",
    "summarize_low_mse_high_risk",
]
