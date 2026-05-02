"""Toy rigid-body domain contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mechanismlens.metrics.physics import boundary_violation, momentum_drift, penetration_violation
from mechanismlens.schema import Finding, Trajectory


@dataclass
class ToyRigidBodyContract:
    """Run simple rigid-body feasibility checks."""

    bounds: Sequence[tuple[float, float]] | dict[str | int, tuple[float, float]] | None = None
    name: str = "toy_rigid_body"

    def check(self, trajectory: Trajectory) -> list[Finding]:
        findings: list[Finding] = []
        _, boundary_findings = boundary_violation(trajectory, self.bounds)
        _, penetration_findings = penetration_violation(trajectory)
        _, momentum_findings = momentum_drift(trajectory)
        findings.extend(boundary_findings)
        findings.extend(penetration_findings)
        findings.extend(momentum_findings)
        return findings

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ToyRigidBodyContract":
        config = config or {}
        return cls(bounds=config.get("bounds"))
