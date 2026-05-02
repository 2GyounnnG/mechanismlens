from mechanismlens import ObjectState, Trajectory
from mechanismlens.metrics.horizon import horizon_amplification, mean_position_error


def test_horizon_error_works() -> None:
    predicted = Trajectory(
        states=[
            [ObjectState("obj", position=[0.0, 0.0])],
            [ObjectState("obj", position=[2.0, 0.0])],
        ]
    )
    ground_truth = Trajectory(
        states=[
            [ObjectState("obj", position=[0.0, 0.0])],
            [ObjectState("obj", position=[1.0, 0.0])],
        ]
    )

    errors = mean_position_error(predicted, ground_truth)
    amplification = horizon_amplification(errors)

    assert errors == [0.0, 1.0]
    assert amplification["h1"] == 0.0
    assert amplification["hfinal"] == 1.0
