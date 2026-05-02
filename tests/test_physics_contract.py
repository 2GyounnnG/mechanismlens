from mechanismlens import ObjectState, Trajectory
from mechanismlens.metrics.cross_layer import semantic_physics_mismatch
from mechanismlens.metrics.physics import boundary_violation


def test_boundary_violation_is_detected() -> None:
    trajectory = Trajectory(states=[[ObjectState("obj", position=[2.0, 0.0], radius=0.2)]])

    metric, findings = boundary_violation(trajectory, bounds=[(-1.0, 1.0), (-1.0, 1.0)])

    assert metric == 1.0
    assert findings
    assert findings[0].category == "physics"


def test_semantic_physics_mismatch_is_detected() -> None:
    trajectory = Trajectory(
        states=[
            [ObjectState("wall", label="static obstacle", position=[0.0, 0.0], radius=1.0)],
            [ObjectState("wall", label="static obstacle", position=[0.1, 0.0], radius=1.0)],
        ]
    )

    findings = semantic_physics_mismatch(trajectory)

    assert any("Static object" in finding.message for finding in findings)
