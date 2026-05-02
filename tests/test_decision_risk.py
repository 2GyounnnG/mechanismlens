from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory
from mechanismlens.examples.benchmark_cases import load_toy_benchmark_cases
from mechanismlens.metrics.decision import (
    decision_risk_findings,
    imagined_real_return_gap,
    uncertainty_on_planned_path,
)


def make_decision_trajectory() -> Trajectory:
    return Trajectory(
        states=[
            [
                ObjectState("agent", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.0, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
            [
                ObjectState("agent", position=[0.6, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.5),
                ObjectState("wall", label="static obstacle", position=[1.1, 0.0], velocity=[0.0, 0.0], mass=5.0, radius=0.5),
            ],
        ]
    )


def test_imagined_real_return_gap_works() -> None:
    metrics = imagined_real_return_gap([1.0, 2.0], [0.5, 0.25])

    assert metrics == {
        "predicted_return": 3.0,
        "realized_return": 0.75,
        "return_gap": 2.25,
    }


def test_high_return_gap_creates_decision_finding() -> None:
    findings = decision_risk_findings(
        predicted_rewards=[1.0, 1.0],
        realized_rewards=[0.0, -0.2],
    )

    assert any(finding.category == "decision" and finding.severity == "high" for finding in findings)


def test_uncertainty_finding_works() -> None:
    metrics = uncertainty_on_planned_path([0.1, 0.9])
    findings = decision_risk_findings(uncertainty=[0.1, 0.9])

    assert metrics["max_uncertainty"] == 0.9
    assert any("uncertainty" in finding.message for finding in findings)


def test_audit_suite_includes_decision_metrics_and_findings() -> None:
    report = AuditSuite().run(
        AuditInput(
            predicted=make_decision_trajectory(),
            predicted_rewards=[0.2, 1.4],
            realized_rewards=[0.2, -0.4],
            uncertainty=[0.1, 0.9],
        )
    )

    assert "decision_return_gap" in report.metrics
    assert "decision_uncertainty" in report.metrics
    assert any(finding.category == "decision" for finding in report.findings)


def test_benchmark_includes_planner_exploit_failure() -> None:
    case_names = [case.name for case in load_toy_benchmark_cases()]

    assert "planner_exploit_failure" in case_names
