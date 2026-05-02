"""Domain contract interface for plugin-style audits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mechanismlens.schema import Finding, Trajectory


class DomainContract(ABC):
    """Base class for domain-specific audit plugins."""

    name: str = "domain_contract"
    description: str = "Domain-specific checks for MechanismLens trajectories."

    @abstractmethod
    def check_trajectory(self, trajectory: Trajectory) -> tuple[dict[str, Any], list[Finding]]:
        """Return metrics and findings for a predicted trajectory."""

    def check_counterfactual(
        self,
        base: Trajectory,
        intervened: Trajectory,
        expected_affected_object_ids: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[Finding]]:
        """Return metrics and findings for base/intervened predicted rollouts."""

        return {}, []
