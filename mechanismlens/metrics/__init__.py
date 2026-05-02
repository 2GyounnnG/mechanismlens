"""Metric functions exposed by MechanismLens."""

from .causal import (
    intervention_locality_score,
    per_object_deviation,
    unexpected_side_effect_findings,
)
from .cross_layer import semantic_physics_mismatch
from .decision import (
    action_switch_rate,
    decision_risk_findings,
    imagined_real_return_gap,
    uncertainty_on_planned_path,
    violation_reward_coupling,
)
from .horizon import horizon_amplification, mean_position_error
from .physics import boundary_violation, momentum_drift, penetration_violation

__all__ = [
    "action_switch_rate",
    "boundary_violation",
    "decision_risk_findings",
    "horizon_amplification",
    "imagined_real_return_gap",
    "intervention_locality_score",
    "mean_position_error",
    "momentum_drift",
    "per_object_deviation",
    "penetration_violation",
    "semantic_physics_mismatch",
    "uncertainty_on_planned_path",
    "unexpected_side_effect_findings",
    "violation_reward_coupling",
]
