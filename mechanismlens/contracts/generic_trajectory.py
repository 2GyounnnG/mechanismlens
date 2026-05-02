"""Generic trajectory contract for framework-independent audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mechanismlens.contracts.base import DomainContract
from mechanismlens.metrics.cross_layer import semantic_physics_mismatch
from mechanismlens.schema import Finding, Trajectory


@dataclass
class GenericTrajectoryContract(DomainContract):
    """Run safe checks that require only object ids, labels, and positions."""

    name: str = "generic_trajectory"
    description: str = "Generic trajectory checks that avoid domain-specific physics assumptions."

    def check_trajectory(self, trajectory: Trajectory) -> tuple[dict[str, Any], list[Finding]]:
        findings: list[Finding] = []
        if not trajectory.states:
            findings.append(
                Finding(
                    severity="high",
                    category="horizon",
                    message="Trajectory contains no states.",
                    details={"contract": self.name},
                )
            )

        cross_layer_findings = semantic_physics_mismatch(trajectory)
        findings.extend(cross_layer_findings)
        metrics: dict[str, Any] = {
            "num_timesteps": len(trajectory.states),
            "num_objects": len(trajectory.object_ids()),
            "cross_layer_mismatch_count": len(cross_layer_findings),
        }
        return metrics, findings

    def check(self, trajectory: Trajectory) -> list[Finding]:
        """Backward-compatible v0.1 contract method."""

        return self.check_trajectory(trajectory)[1]
