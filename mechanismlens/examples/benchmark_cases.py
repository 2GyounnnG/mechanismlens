"""Predefined deterministic toy benchmark cases."""

from __future__ import annotations

from mechanismlens import AuditInput, ObjectState, Trajectory
from mechanismlens.benchmark import BenchmarkCase


def clean_rollout() -> BenchmarkCase:
    trajectory = Trajectory(
        states=[
            [
                ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("wall", label="static obstacle", position=[3.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4),
            ],
            [
                ObjectState("ball", label="rigid", position=[0.5, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("wall", label="static obstacle", position=[3.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4),
            ],
        ],
        metadata={"case": "clean_rollout"},
    )
    return BenchmarkCase(
        name="clean_rollout",
        description="A small rollout with no expected findings.",
        audit_input=AuditInput(predicted=trajectory),
    )


def penetration_failure() -> BenchmarkCase:
    trajectory = Trajectory(
        states=[
            [
                ObjectState("a", label="rigid", position=[0.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", label="rigid", position=[0.5, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
            ]
        ],
        metadata={"case": "penetration_failure"},
    )
    return BenchmarkCase(
        name="penetration_failure",
        description="Two rigid objects overlap.",
        audit_input=AuditInput(predicted=trajectory),
    )


def static_object_mismatch() -> BenchmarkCase:
    trajectory = Trajectory(
        states=[
            [ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4)],
            [ObjectState("wall", label="static obstacle", position=[2.2, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.4)],
        ],
        metadata={"case": "static_object_mismatch"},
    )
    return BenchmarkCase(
        name="static_object_mismatch",
        description="An object labeled static moves over time.",
        audit_input=AuditInput(predicted=trajectory),
    )


def counterfactual_side_effect() -> BenchmarkCase:
    base = Trajectory(
        states=[
            [ObjectState("A", position=[0.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
        ],
        metadata={"case": "counterfactual_base"},
    )
    intervened = Trajectory(
        states=[
            [ObjectState("A", position=[0.5, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.5, 0.0]), ObjectState("D", position=[10.7, 0.0])],
        ],
        metadata={"case": "counterfactual_intervened"},
    )
    return BenchmarkCase(
        name="counterfactual_side_effect",
        description="Object A is targeted but unrelated object D changes.",
        audit_input=AuditInput(
            predicted=intervened,
            predicted_base=base,
            predicted_intervened=intervened,
            expected_affected_object_ids=["A"],
            intervention_description="Move object A to the right.",
        ),
    )


def combined_failure() -> BenchmarkCase:
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
        metadata={"case": "combined_base"},
    )
    predicted = Trajectory(
        states=[
            [
                ObjectState("A", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[0.8, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
            [
                ObjectState("A", label="rigid", position=[0.5, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("D", label="static obstacle", position=[1.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.6),
            ],
        ],
        metadata={"case": "combined_predicted"},
    )
    return BenchmarkCase(
        name="combined_failure",
        description="Penetration, momentum drift, static-object mismatch, and side effect.",
        audit_input=AuditInput(
            predicted=predicted,
            predicted_base=base,
            predicted_intervened=predicted,
            expected_affected_object_ids=["A"],
            intervention_description="Move object A while D should remain fixed.",
        ),
    )


def planner_exploit_failure() -> BenchmarkCase:
    trajectory = Trajectory(
        states=[
            [
                ObjectState("agent", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("agent", label="rigid", position=[0.6, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.1, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("agent", label="rigid", position=[0.8, 0.0], velocity=[2.5, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.3, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
        ],
        metadata={"case": "planner_exploit_failure"},
    )
    return BenchmarkCase(
        name="planner_exploit_failure",
        description="Planner imagines high reward on invalid, uncertain rollout states.",
        audit_input=AuditInput(
            predicted=trajectory,
            planned_actions=[
                {"action": "move_right"},
                {"action": "push_through_wall"},
                {"action": "continue"},
            ],
            predicted_rewards=[0.2, 1.4, 1.6],
            realized_rewards=[0.2, -0.2, -0.4],
            uncertainty=[0.1, 0.9, 1.1],
            planner_metadata={"planner": "toy_greedy", "objective": "predicted_reward"},
        ),
    )


def load_toy_benchmark_cases() -> list[BenchmarkCase]:
    """Return the deterministic toy benchmark suite."""

    return [
        clean_rollout(),
        penetration_failure(),
        static_object_mismatch(),
        counterfactual_side_effect(),
        combined_failure(),
        planner_exploit_failure(),
    ]
