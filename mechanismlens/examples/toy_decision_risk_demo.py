"""Runnable toy decision-risk audit demo.

Run with:
    python -m mechanismlens.examples.toy_decision_risk_demo
"""

from __future__ import annotations

from pathlib import Path

from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory

REPORT_MARKDOWN_PATH = Path("examples/reports/toy_decision_risk_audit.md")
REPORT_JSON_PATH = Path("examples/reports/toy_decision_risk_audit.json")


def build_predicted_rollout() -> Trajectory:
    """Construct a rollout where reward-seeking exploits invalid states."""

    return Trajectory(
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
        metadata={"name": "decision_risk_predicted"},
    )


def main() -> None:
    audit_input = AuditInput(
        predicted=build_predicted_rollout(),
        planned_actions=[
            {"action": "move_right"},
            {"action": "push_through_wall"},
            {"action": "continue"},
        ],
        predicted_rewards=[0.2, 1.4, 1.6],
        realized_rewards=[0.2, -0.2, -0.4],
        uncertainty=[0.1, 0.9, 1.1],
        planner_metadata={"planner": "toy_greedy", "objective": "predicted_reward"},
    )
    report = AuditSuite(bounds=[(-1.0, 3.0), (-1.0, 1.0)]).run(audit_input)
    print(report.to_markdown())
    REPORT_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_markdown(REPORT_MARKDOWN_PATH)
    report.save_json(REPORT_JSON_PATH)


if __name__ == "__main__":
    main()
