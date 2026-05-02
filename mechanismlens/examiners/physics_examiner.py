"""Small physics examiner wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mechanismlens.metrics.physics import boundary_violation, momentum_drift, penetration_violation
from mechanismlens.schema import Finding, Trajectory


@dataclass
class PhysicsExaminer:
    bounds: Sequence[tuple[float, float]] | dict[str | int, tuple[float, float]] | None = None

    def examine(self, trajectory: Trajectory) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(boundary_violation(trajectory, self.bounds)[1])
        findings.extend(penetration_violation(trajectory)[1])
        findings.extend(momentum_drift(trajectory)[1])
        return findings
