"""Small intervention-locality metric demo."""

from __future__ import annotations

from mechanismlens import ObjectState, Trajectory
from mechanismlens.metrics.causal import intervention_locality_score


def main() -> None:
    base = Trajectory(
        states=[[ObjectState("a", position=[0.0, 0.0]), ObjectState("b", position=[1.0, 0.0])]]
    )
    intervened = Trajectory(
        states=[[ObjectState("a", position=[0.5, 0.0]), ObjectState("b", position=[1.05, 0.0])]]
    )
    print(intervention_locality_score(base, intervened, {"a"}))


if __name__ == "__main__":
    main()
