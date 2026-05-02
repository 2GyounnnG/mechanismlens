"""Deterministic synthetic cases for mechanism-mismatch experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mechanismlens import AuditInput, ObjectState, Trajectory
from mechanismlens.benchmark import BenchmarkCase


@dataclass(frozen=True)
class SyntheticCaseSpec:
    """Ground-truth metadata for one synthetic experiment case."""

    case_id: str
    injected_failures: list[str]
    severity: str
    has_downstream_failure: bool
    seed: int

    def to_labels(self) -> dict[str, Any]:
        """Return JSON-friendly ground-truth labels."""

        return {
            "case_id": self.case_id,
            "injected_failures": self.injected_failures,
            "severity": self.severity,
            "has_downstream_failure": self.has_downstream_failure,
            "seed": self.seed,
        }


def _jitter(seed: int, scale: float = 0.03) -> float:
    return random.Random(seed).uniform(-scale, scale)


def _case(case_id: str, description: str, audit_input: AuditInput, spec: SyntheticCaseSpec) -> tuple[BenchmarkCase, dict[str, Any]]:
    return (
        BenchmarkCase(
            name=case_id,
            description=description,
            audit_input=audit_input,
            contract_name="toy_rigid_body",
        ),
        spec.to_labels(),
    )


def generate_clean_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a clean rollout with no injected mechanism failure."""

    offset = _jitter(seed)
    predicted = Trajectory(
        states=[
            [
                ObjectState("ball", label="rigid", position=[0.0 + offset, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("wall", label="static obstacle", position=[3.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4),
            ],
            [
                ObjectState("ball", label="rigid", position=[0.5 + offset, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("wall", label="static obstacle", position=[3.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4),
            ],
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(f"clean_{seed}", [], "none", False, seed)
    return _case(spec.case_id, "Clean synthetic rollout.", AuditInput(predicted=predicted), spec)


def generate_physics_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate an object-penetration physics failure."""

    shift = _jitter(seed)
    predicted = Trajectory(
        states=[
            [
                ObjectState("a", label="rigid", position=[0.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", label="rigid", position=[0.5 + shift, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
            ]
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(f"physics_{seed}", ["physics"], "high", True, seed)
    return _case(spec.case_id, "Injected object penetration.", AuditInput(predicted=predicted), spec)


def generate_cross_layer_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a static-label cross-layer mismatch."""

    movement = 0.18 + abs(_jitter(seed))
    predicted = Trajectory(
        states=[
            [ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4)],
            [ObjectState("wall", label="static obstacle", position=[2.0 + movement, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4)],
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(f"cross_layer_{seed}", ["cross_layer"], "medium", False, seed)
    return _case(spec.case_id, "Injected static-object semantic/physics mismatch.", AuditInput(predicted=predicted), spec)


def generate_causal_side_effect_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a counterfactual with an unexpected side effect."""

    side_effect = 0.65 + abs(_jitter(seed))
    base = Trajectory(
        states=[
            [ObjectState("A", position=[0.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
        ],
        metadata={"synthetic_seed": seed, "role": "base"},
    )
    intervened = Trajectory(
        states=[
            [ObjectState("A", position=[0.4, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.5, 0.0]), ObjectState("D", position=[10.0 + side_effect, 0.0])],
        ],
        metadata={"synthetic_seed": seed, "role": "intervened"},
    )
    spec = SyntheticCaseSpec(f"causal_{seed}", ["causal"], "medium", True, seed)
    return _case(
        spec.case_id,
        "Injected counterfactual side effect.",
        AuditInput(
            predicted=intervened,
            predicted_base=base,
            predicted_intervened=intervened,
            expected_affected_object_ids=["A"],
            intervention_description="Move A; D should remain fixed.",
        ),
        spec,
    )


def generate_decision_risk_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a planner exploit case with high imagined reward and low realized reward."""

    reward_boost = abs(_jitter(seed, scale=0.1))
    predicted = Trajectory(
        states=[
            [
                ObjectState("agent", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("agent", label="rigid", position=[0.6, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.1, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(f"decision_{seed}", ["decision"], "high", True, seed)
    return _case(
        spec.case_id,
        "Injected planner exploit / decision-risk case.",
        AuditInput(
            predicted=predicted,
            planned_actions=[{"action": "move_right"}, {"action": "push_through_wall"}],
            predicted_rewards=[0.4, 1.5 + reward_boost],
            realized_rewards=[0.1, -0.6],
            uncertainty=[0.2, 0.95],
            planner_metadata={"planner": "synthetic_greedy"},
        ),
        spec,
    )


def generate_combined_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a case with physics, cross-layer, causal, and decision failures."""

    base = Trajectory(
        states=[
            [
                ObjectState("A", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
            [
                ObjectState("A", label="rigid", position=[0.5, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
        ],
        metadata={"synthetic_seed": seed, "role": "base"},
    )
    predicted = Trajectory(
        states=[
            [
                ObjectState("A", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[0.8, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
            [
                ObjectState("A", label="rigid", position=[0.5, 0.0], velocity=[2.2, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[1.1, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
        ],
        metadata={"synthetic_seed": seed, "role": "predicted"},
    )
    spec = SyntheticCaseSpec(
        f"combined_{seed}",
        ["physics", "cross_layer", "causal", "decision"],
        "high",
        True,
        seed,
    )
    return _case(
        spec.case_id,
        "Combined synthetic mechanism mismatch.",
        AuditInput(
            predicted=predicted,
            predicted_base=base,
            predicted_intervened=predicted,
            expected_affected_object_ids=["A"],
            predicted_rewards=[0.5, 1.8],
            realized_rewards=[0.0, -0.8],
            uncertainty=[0.2, 1.0],
            intervention_description="A should move; D should remain fixed.",
        ),
        spec,
    )
