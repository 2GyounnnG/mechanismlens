from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory


def test_audit_suite_returns_report_with_findings() -> None:
    predicted = Trajectory(
        states=[
            [
                ObjectState("a", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", label="static obstacle", position=[0.8, 0.0], velocity=[0.0, 0.0], mass=2.0, radius=0.6),
            ],
            [
                ObjectState("a", label="rigid", position=[0.2, 0.0], velocity=[2.0, 0.0], mass=1.0, radius=0.6),
                ObjectState("b", label="static obstacle", position=[0.9, 0.0], velocity=[0.0, 0.0], mass=2.0, radius=0.6),
            ],
        ]
    )

    report = AuditSuite(bounds=[(-1.0, 1.0), (-1.0, 1.0)]).run(AuditInput(predicted=predicted))

    assert report.findings
    assert report.overall_risk in {"medium", "high"}
    assert "penetration_violation" in report.metrics
    assert "MechanismLens Audit Report" in report.to_markdown()
