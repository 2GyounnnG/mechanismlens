"""Deterministic synthetic cases for mechanism-mismatch experiments."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mechanismlens import AuditInput, ObjectState, Trajectory
from mechanismlens.benchmark import BenchmarkCase


@dataclass(frozen=True)
class SyntheticCaseSpec:
    """Ground-truth metadata for one synthetic experiment case."""

    case_id: str
    failure_type: str
    injected_failures: list[str]
    severity: str
    has_downstream_failure: bool
    expected_risk: str
    seed: int
    notes: str = ""

    def to_labels(self, *, expected_low_mse: bool) -> dict[str, Any]:
        """Return JSON-friendly ground-truth labels."""

        return {
            "case_id": self.case_id,
            "failure_type": self.failure_type,
            "injected_failures": self.injected_failures,
            "severity": self.severity,
            "has_downstream_failure": self.has_downstream_failure,
            "expected_risk": self.expected_risk,
            "seed": self.seed,
            "expected_low_mse": expected_low_mse,
            "notes": self.notes,
        }


def generate_clean_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a clean rollout with no mechanism failure."""

    offset = _jitter(seed)
    predicted = _clean_trajectory(offset=offset)
    ground_truth = _clean_trajectory(offset=offset)
    spec = SyntheticCaseSpec(
        case_id=f"clean_{seed}",
        failure_type="clean",
        injected_failures=[],
        severity="none",
        has_downstream_failure=False,
        expected_risk="low",
        seed=seed,
    )
    return _case(
        spec,
        "Clean synthetic rollout.",
        AuditInput(predicted=predicted, ground_truth=ground_truth),
        True,
    )


def generate_prediction_error_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate moderate prediction error without mechanism violation."""

    offset = 0.35 + abs(_jitter(seed))
    ground_truth = _clean_trajectory(offset=0.0)
    predicted = _clean_trajectory(offset=offset)
    spec = SyntheticCaseSpec(
        case_id=f"prediction_error_{seed}",
        failure_type="prediction_error",
        injected_failures=[],
        severity="medium",
        has_downstream_failure=False,
        expected_risk="low",
        seed=seed,
        notes="Moderate MSE without contract violation.",
    )
    return _case(
        spec,
        "Moderate position error without mechanism mismatch.",
        AuditInput(predicted=predicted, ground_truth=ground_truth),
        False,
    )


def generate_physics_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate an object-penetration physics failure."""

    shift = _jitter(seed)
    predicted = Trajectory(
        states=[
            [
                ObjectState(
                    "a",
                    label="rigid",
                    position=[0.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
                ObjectState(
                    "b",
                    label="rigid",
                    position=[0.5 + shift, 0.0],
                    velocity=[0.0, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
            ]
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(
        f"physics_{seed}", "physics", ["physics"], "high", True, "high", seed
    )
    return _case(spec, "Injected object penetration.", AuditInput(predicted=predicted), True)


def generate_cross_layer_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a static-label cross-layer mismatch."""

    movement = 0.18 + abs(_jitter(seed))
    predicted = Trajectory(
        states=[
            [
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[2.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.4,
                )
            ],
            [
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[2.0 + movement, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.4,
                )
            ],
        ],
        metadata={"synthetic_seed": seed},
    )
    spec = SyntheticCaseSpec(
        f"cross_layer_{seed}", "cross_layer", ["cross_layer"], "medium", False, "medium", seed
    )
    return _case(
        spec,
        "Injected static-object semantic/physics mismatch.",
        AuditInput(predicted=predicted),
        True,
    )


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
            [
                ObjectState("A", position=[1.5, 0.0]),
                ObjectState("D", position=[10.0 + side_effect, 0.0]),
            ],
        ],
        metadata={"synthetic_seed": seed, "role": "intervened"},
    )
    spec = SyntheticCaseSpec(
        f"causal_{seed}", "causal", ["causal"], "medium", True, "medium", seed
    )
    return _case(
        spec,
        "Injected counterfactual side effect.",
        AuditInput(
            predicted=intervened,
            predicted_base=base,
            predicted_intervened=intervened,
            expected_affected_object_ids=["A"],
            intervention_description="Move A; D should remain fixed.",
        ),
        True,
    )


def generate_decision_risk_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a planner exploit case with high imagined reward and low realized reward."""

    reward_boost = abs(_jitter(seed, scale=0.1))
    predicted = _decision_risk_trajectory(seed)
    spec = SyntheticCaseSpec(
        f"decision_{seed}", "decision", ["decision"], "high", True, "high", seed
    )
    return _case(
        spec,
        "Injected planner exploit / decision-risk case.",
        AuditInput(
            predicted=predicted,
            planned_actions=[{"action": "move_right"}, {"action": "push_through_wall"}],
            predicted_rewards=[0.4, 1.5 + reward_boost],
            realized_rewards=[0.1, -0.6],
            uncertainty=[0.2, 0.95],
            planner_metadata={"planner": "synthetic_greedy"},
        ),
        True,
    )


def generate_low_mse_high_risk_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate low prediction error with high mechanism risk."""

    predicted = _decision_risk_trajectory(seed)
    ground_truth = _decision_risk_trajectory(seed)
    spec = SyntheticCaseSpec(
        case_id=f"low_mse_high_risk_{seed}",
        failure_type="low_mse_high_risk",
        injected_failures=["physics", "cross_layer", "decision"],
        severity="high",
        has_downstream_failure=True,
        expected_risk="high",
        seed=seed,
        notes="Prediction positions match ground truth, but mechanism contracts fail.",
    )
    return _case(
        spec,
        "Low MSE but high mechanism risk.",
        AuditInput(
            predicted=predicted,
            ground_truth=ground_truth,
            predicted_rewards=[0.2, 1.6],
            realized_rewards=[0.1, -0.7],
            uncertainty=[0.2, 0.95],
            planner_metadata={"planner": "synthetic_greedy"},
        ),
        True,
    )


def generate_combined_failure_case(seed: int) -> tuple[BenchmarkCase, dict[str, Any]]:
    """Generate a case with physics, cross-layer, causal, and decision failures."""

    base = Trajectory(
        states=[
            [
                ObjectState(
                    "A",
                    label="rigid",
                    position=[0.0, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
                ObjectState(
                    "D",
                    label="static obstacle",
                    position=[2.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.6,
                ),
            ],
            [
                ObjectState(
                    "A",
                    label="rigid",
                    position=[0.5, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
                ObjectState(
                    "D",
                    label="static obstacle",
                    position=[2.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.6,
                ),
            ],
        ],
        metadata={"synthetic_seed": seed, "role": "base"},
    )
    predicted = Trajectory(
        states=[
            [
                ObjectState(
                    "A",
                    label="rigid",
                    position=[0.0, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
                ObjectState(
                    "D",
                    label="static obstacle",
                    position=[0.8, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.6,
                ),
            ],
            [
                ObjectState(
                    "A",
                    label="rigid",
                    position=[0.5, 0.0],
                    velocity=[2.2, 0.0],
                    mass=1.0,
                    radius=0.6,
                ),
                ObjectState(
                    "D",
                    label="static obstacle",
                    position=[1.1, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.6,
                ),
            ],
        ],
        metadata={"synthetic_seed": seed, "role": "predicted"},
    )
    spec = SyntheticCaseSpec(
        f"combined_{seed}",
        "combined",
        ["physics", "cross_layer", "causal", "decision"],
        "high",
        True,
        "high",
        seed,
    )
    return _case(
        spec,
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
        True,
    )


def generate_synthetic_suite(
    n_per_type: int = 20,
    seed: int = 0,
) -> list[tuple[BenchmarkCase, dict[str, Any]]]:
    """Generate all synthetic case types deterministically."""

    generators: tuple[tuple[str, Callable[[int], tuple[BenchmarkCase, dict[str, Any]]]], ...] = (
        ("clean", generate_clean_case),
        ("prediction_error", generate_prediction_error_case),
        ("physics", generate_physics_failure_case),
        ("cross_layer", generate_cross_layer_failure_case),
        ("causal", generate_causal_side_effect_case),
        ("decision", generate_decision_risk_case),
        ("low_mse_high_risk", generate_low_mse_high_risk_case),
        ("combined", generate_combined_failure_case),
    )
    cases: list[tuple[BenchmarkCase, dict[str, Any]]] = []
    for type_index, (_name, generator) in enumerate(generators):
        for case_index in range(n_per_type):
            cases.append(generator(seed + type_index * 10_000 + case_index))
    return cases


def _jitter(seed: int, scale: float = 0.03) -> float:
    return random.Random(seed).uniform(-scale, scale)


def _case(
    spec: SyntheticCaseSpec,
    description: str,
    audit_input: AuditInput,
    expected_low_mse: bool,
) -> tuple[BenchmarkCase, dict[str, Any]]:
    return (
        BenchmarkCase(
            name=spec.case_id,
            description=description,
            audit_input=audit_input,
            contract_name="toy_rigid_body",
        ),
        spec.to_labels(expected_low_mse=expected_low_mse),
    )


def _clean_trajectory(offset: float = 0.0) -> Trajectory:
    return Trajectory(
        states=[
            [
                ObjectState(
                    "ball",
                    label="rigid",
                    position=[0.0 + offset, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.2,
                ),
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[3.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.4,
                ),
            ],
            [
                ObjectState(
                    "ball",
                    label="rigid",
                    position=[0.5 + offset, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.2,
                ),
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[3.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.4,
                ),
            ],
        ]
    )


def _decision_risk_trajectory(seed: int) -> Trajectory:
    return Trajectory(
        states=[
            [
                ObjectState(
                    "agent",
                    label="rigid",
                    position=[0.0, 0.0],
                    velocity=[1.0, 0.0],
                    mass=1.0,
                    radius=0.5,
                ),
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[1.0, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.5,
                ),
            ],
            [
                ObjectState(
                    "agent",
                    label="rigid",
                    position=[0.6, 0.0],
                    velocity=[2.0, 0.0],
                    mass=1.0,
                    radius=0.5,
                ),
                ObjectState(
                    "wall",
                    label="static obstacle",
                    position=[1.1, 0.0],
                    velocity=[0.0, 0.0],
                    mass=5.0,
                    radius=0.5,
                ),
            ],
        ],
        metadata={"synthetic_seed": seed},
    )
