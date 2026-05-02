"""Toy rigid-body domain contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mechanismlens.contracts.base import DomainContract
from mechanismlens.metrics.cross_layer import semantic_physics_mismatch
from mechanismlens.metrics.physics import boundary_violation, momentum_drift, penetration_violation
from mechanismlens.schema import Finding, Trajectory


@dataclass
class ToyRigidBodyContract(DomainContract):
    """Run simple rigid-body and semantic/physics consistency checks."""

    bounds: Sequence[tuple[float, float]] | dict[str | int, tuple[float, float]] | None = None
    name: str = "toy_rigid_body"
    description: str = "Toy rigid-body checks for boundaries, penetration, momentum, and labels."

    def check_trajectory(self, trajectory: Trajectory) -> tuple[dict[str, Any], list[Finding]]:
        boundary_metric, boundary_findings = boundary_violation(trajectory, self.bounds)
        penetration_metric, penetration_findings = penetration_violation(trajectory)
        momentum_metric, momentum_findings = momentum_drift(trajectory)
        cross_layer_findings = semantic_physics_mismatch(trajectory)

        metrics: dict[str, Any] = {
            "boundary_violation": boundary_metric,
            "penetration_violation": penetration_metric,
            "momentum_drift": momentum_metric,
            "cross_layer_mismatch_count": len(cross_layer_findings),
        }
        findings = [
            *boundary_findings,
            *penetration_findings,
            *momentum_findings,
            *cross_layer_findings,
        ]
        return metrics, findings

    def check(self, trajectory: Trajectory) -> list[Finding]:
        """Backward-compatible v0.1 contract method."""

        return self.check_trajectory(trajectory)[1]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ToyRigidBodyContract":
        config = config or {}
        return cls(bounds=config.get("bounds"))
