"""Cross-layer examiner wrapper."""

from __future__ import annotations

from mechanismlens.metrics.cross_layer import semantic_physics_mismatch
from mechanismlens.schema import Finding, Trajectory


class CrossLayerExaminer:
    def examine(self, trajectory: Trajectory) -> list[Finding]:
        return semantic_physics_mismatch(trajectory)
