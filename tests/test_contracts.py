from mechanismlens import AuditInput, AuditReport, AuditSuite, Finding, ObjectState, Trajectory
from mechanismlens.contracts import GenericTrajectoryContract, ToyRigidBodyContract


def test_default_audit_suite_still_uses_toy_contract() -> None:
    trajectory = Trajectory(
        states=[
            [
                ObjectState("a", position=[0.0, 0.0], radius=0.6),
                ObjectState("b", position=[0.5, 0.0], radius=0.6),
            ]
        ]
    )

    report = AuditSuite().run(AuditInput(predicted=trajectory))

    assert "penetration_violation" in report.metrics
    assert any(finding.category == "physics" for finding in report.findings)


def test_toy_rigid_body_contract_returns_metrics_and_findings() -> None:
    trajectory = Trajectory(
        states=[
            [
                ObjectState("a", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", position=[0.5, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
            ],
            [
                ObjectState("a", position=[0.2, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", position=[0.7, 0.0], velocity=[0.0, 0.0], mass=1.0, radius=0.6),
            ],
        ]
    )

    metrics, findings = ToyRigidBodyContract(bounds=[(-1.0, 1.0), (-1.0, 1.0)]).check_trajectory(
        trajectory
    )

    assert metrics["penetration_violation"] > 0.0
    assert "momentum_drift" in metrics
    assert findings


def test_generic_trajectory_contract_does_not_crash_with_minimal_trajectory() -> None:
    trajectory = Trajectory(states=[[ObjectState("minimal", position=[0.0, 0.0])]])

    metrics, findings = GenericTrajectoryContract().check_trajectory(trajectory)

    assert metrics["num_timesteps"] == 1
    assert metrics["num_objects"] == 1
    assert findings == []


def test_report_save_method_creates_requested_file(tmp_path) -> None:
    report = AuditReport(
        overall_risk="low",
        findings=[Finding(severity="low", category="semantic", message="demo")],
        metrics={"ok": True},
    )
    path = tmp_path / "reports" / "audit_report.md"
    path.parent.mkdir()

    report.save_markdown(path)

    assert path.exists()
    assert "MechanismLens Audit Report" in path.read_text(encoding="utf-8")
