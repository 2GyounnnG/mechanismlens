"""Runnable toy rollout audit demo.

Run with:
    python -m mechanismlens.examples.toy_rollout_demo
"""

from __future__ import annotations

from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory


def build_ground_truth() -> Trajectory:
    """Construct a simple two-object non-penetrating ground-truth rollout."""

    return Trajectory(
        states=[
            [
                ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
            [
                ObjectState("ball", label="rigid", position=[0.5, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
            [
                ObjectState("ball", label="rigid", position=[1.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
        ],
        metadata={"name": "ground_truth"},
    )


def build_predicted() -> Trajectory:
    """Construct a predicted rollout with v0.1 audit failures."""

    return Trajectory(
        states=[
            [
                ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
            [
                ObjectState("ball", label="rigid", position=[1.65, 0.0], velocity=[3.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.05, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
            [
                ObjectState("ball", label="rigid", position=[1.9, 0.0], velocity=[4.0, 0.0], mass=1.0, radius=0.4),
                ObjectState("wall", label="static obstacle", position=[2.2, 0.0], velocity=[0.0, 0.0], mass=10.0, radius=0.5),
            ],
        ],
        metadata={"name": "predicted_with_failures"},
    )


def main() -> None:
    audit_input = AuditInput(predicted=build_predicted(), ground_truth=build_ground_truth())
    report = AuditSuite(bounds=[(-1.0, 3.0), (-1.0, 1.0)]).run(audit_input)
    print(report.to_markdown())
    report.save_markdown("audit_report.md")


if __name__ == "__main__":
    main()
