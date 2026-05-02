"""Domain contract protocol."""

from __future__ import annotations

from typing import Protocol

from mechanismlens.schema import Finding, Trajectory


class DomainContract(Protocol):
    """Optional domain-specific checks over a trajectory."""

    name: str

    def check(self, trajectory: Trajectory) -> list[Finding]:
        """Return findings for contract violations."""
        ...
