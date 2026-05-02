from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory
from mechanismlens.metrics.causal import per_object_deviation, unexpected_side_effect_findings


def make_counterfactual_pair() -> tuple[Trajectory, Trajectory]:
    base = Trajectory(
        states=[
            [ObjectState("A", position=[0.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.0, 0.0]), ObjectState("D", position=[10.0, 0.0])],
        ]
    )
    intervened = Trajectory(
        states=[
            [ObjectState("A", position=[0.5, 0.0]), ObjectState("D", position=[10.0, 0.0])],
            [ObjectState("A", position=[1.5, 0.0]), ObjectState("D", position=[10.8, 0.0])],
        ]
    )
    return base, intervened


def test_per_object_deviation_returns_expected_values() -> None:
    base, intervened = make_counterfactual_pair()

    deviations = per_object_deviation(base, intervened)

    assert deviations["A"] == 0.5
    assert deviations["D"] == 0.40000000000000036


def test_unexpected_side_effect_is_detected() -> None:
    base, intervened = make_counterfactual_pair()

    findings = unexpected_side_effect_findings(base, intervened, ["A"], threshold=0.1)

    assert len(findings) == 1
    assert findings[0].category == "causal"
    assert findings[0].severity == "medium"
    assert findings[0].details["object_id"] == "D"


def test_audit_suite_includes_causal_findings_for_counterfactual_inputs() -> None:
    base, intervened = make_counterfactual_pair()

    report = AuditSuite().run(
        AuditInput(
            predicted=intervened,
            predicted_base=base,
            predicted_intervened=intervened,
            expected_affected_object_ids=["A"],
            intervention_description="Move A to the right.",
        )
    )

    assert "counterfactual_locality" in report.metrics
    assert any(finding.category == "causal" for finding in report.findings)
    assert report.metrics["counterfactual_locality"]["per_object_deviation"]["A"] == 0.5
