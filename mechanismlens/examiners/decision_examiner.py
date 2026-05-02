"""Decision-risk examiner placeholder for v0.1."""

from __future__ import annotations

from mechanismlens.schema import Finding, Trajectory


class DecisionExaminer:
    def examine(self, trajectory: Trajectory) -> list[Finding]:
        if not trajectory.actions:
            return []
        return [
            Finding(
                severity="low",
                category="decision",
                message="Planning actions are present but v0.1 only records decision context.",
                details={"num_actions": len(trajectory.actions)},
            )
        ]
