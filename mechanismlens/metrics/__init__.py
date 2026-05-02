"""Metric functions exposed by MechanismLens."""

from .causal import (
    intervention_locality_score,
    per_object_deviation,
    unexpected_side_effect_findings,
)
from .cross_layer import semantic_physics_mismatch
from .horizon import horizon_amplification, mean_position_error
from .physics import boundary_violation, momentum_drift, penetration_violation

__all__ = [
    "boundary_violation",
    "horizon_amplification",
    "intervention_locality_score",
    "mean_position_error",
    "momentum_drift",
    "per_object_deviation",
    "penetration_violation",
    "semantic_physics_mismatch",
    "unexpected_side_effect_findings",
]
