"""Runnable counterfactual audit demo.

Run with:
    python -m mechanismlens.examples.toy_counterfactual_demo
"""

from __future__ import annotations

from pathlib import Path

from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory

REPORT_MARKDOWN_PATH = Path("examples/reports/toy_counterfactual_audit.md")
REPORT_JSON_PATH = Path("examples/reports/toy_counterfactual_audit.json")


def build_base_rollout() -> Trajectory:
    """Construct a base prediction with three objects."""

    return Trajectory(
        states=[
            [
                ObjectState("A", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("B", label="rigid", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("D", label="static obstacle", position=[10.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("A", label="rigid", position=[1.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("B", label="rigid", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("D", label="static obstacle", position=[10.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
        ],
        metadata={"name": "base"},
    )


def build_intervened_rollout() -> Trajectory:
    """Construct an intervened prediction with one intended and one unintended effect."""

    return Trajectory(
        states=[
            [
                ObjectState("A", label="rigid", position=[0.4, 0.0], velocity=[1.2, 0.0], mass=1.0, radius=0.2),
                ObjectState("B", label="rigid", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("D", label="static obstacle", position=[10.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("A", label="rigid", position=[1.6, 0.0], velocity=[1.2, 0.0], mass=1.0, radius=0.2),
                ObjectState("B", label="rigid", position=[2.0, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.2),
                ObjectState("D", label="static obstacle", position=[10.7, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
        ],
        metadata={"name": "intervened"},
    )


def main() -> None:
    base = build_base_rollout()
    intervened = build_intervened_rollout()
    audit_input = AuditInput(
        predicted=intervened,
        predicted_base=base,
        predicted_intervened=intervened,
        expected_affected_object_ids=["A"],
        intervention_description="Apply a rightward impulse to object A.",
    )
    report = AuditSuite(bounds=[(-1.0, 12.0), (-1.0, 1.0)]).run(audit_input)
    print(report.to_markdown())
    REPORT_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_markdown(REPORT_MARKDOWN_PATH)
    report.save_json(REPORT_JSON_PATH)


if __name__ == "__main__":
    main()
