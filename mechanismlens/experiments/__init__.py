"""Synthetic experiment utilities for MechanismLens."""

from .analysis import (
    compute_category_confusion,
    compute_detection_summary,
    compute_risk_failure_correlation,
    extract_case_record,
    severity_weighted_risk_score,
)
from .synthetic_generator import (
    SyntheticCaseSpec,
    generate_causal_side_effect_case,
    generate_clean_case,
    generate_combined_failure_case,
    generate_cross_layer_failure_case,
    generate_decision_risk_case,
    generate_physics_failure_case,
)

__all__ = [
    "SyntheticCaseSpec",
    "compute_category_confusion",
    "compute_detection_summary",
    "compute_risk_failure_correlation",
    "extract_case_record",
    "generate_causal_side_effect_case",
    "generate_clean_case",
    "generate_combined_failure_case",
    "generate_cross_layer_failure_case",
    "generate_decision_risk_case",
    "generate_physics_failure_case",
    "severity_weighted_risk_score",
]
