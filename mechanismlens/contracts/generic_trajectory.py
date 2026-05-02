"""Generic trajectory contract kept intentionally small for v0.1."""

from __future__ import annotations

from dataclasses import dataclass

from mechanismlens.schema import Finding, Trajectory


@dataclass
class GenericTrajectoryContract:
    """Check that a trajectory has at least one timestep."""

    name: str = "generic_trajectory"

    def check(self, trajectory: Trajectory) -> list[Finding]:
        if trajectory.states:
            return []
        return [
            Finding(
                severity="high",
                category="horizon",
                message="Trajectory contains no states.",
                details={"contract": self.name},
            )
        ]
