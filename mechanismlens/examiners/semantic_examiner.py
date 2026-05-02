"""Semantic examiner placeholder for v0.1."""

from __future__ import annotations

from mechanismlens.schema import Finding, Trajectory


class SemanticExaminer:
    def examine(self, trajectory: Trajectory) -> list[Finding]:
        return [
            Finding(
                severity="low",
                category="semantic",
                message=f"Trajectory has {len(trajectory.object_ids())} tracked object ids.",
                details={"object_ids": sorted(trajectory.object_ids())},
            )
        ]
