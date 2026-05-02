"""Serializable audit schema for MechanismLens v0.1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

Severity = Literal["low", "medium", "high"]
Category = Literal["semantic", "causal", "physics", "cross_layer", "decision", "horizon"]
Risk = Literal["low", "medium", "high"]


def as_vector(value: Sequence[float] | Any | None) -> list[float]:
    """Convert list-like or NumPy-like values to a JSON-friendly float vector."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def vector_distance(left: Sequence[float] | Any, right: Sequence[float] | Any) -> float:
    """Euclidean distance between two list-like vectors."""

    lhs = as_vector(left)
    rhs = as_vector(right)
    if len(lhs) != len(rhs):
        return float("inf")
    return sum((a - b) ** 2 for a, b in zip(lhs, rhs, strict=True)) ** 0.5


@dataclass(init=False)
class ObjectState:
    """State for one object at one trajectory timestep."""

    object_id: str
    label: str | None = None
    position: Sequence[float] | Any = field(default_factory=list)
    velocity: Sequence[float] | Any | None = None
    mass: float | None = None
    radius: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        object_id: str,
        label: str | None = None,
        position: Sequence[float] | Any | None = None,
        velocity: Sequence[float] | Any | None = None,
        mass: float | None = None,
        radius: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if position is None and label is not None and not isinstance(label, str):
            position = label
            label = None
        if position is None:
            raise ValueError("ObjectState requires a position vector")
        self.object_id = object_id
        self.label = label
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.attributes = attributes or {}

    def position_vector(self) -> list[float]:
        return as_vector(self.position)

    def velocity_vector(self) -> list[float]:
        return as_vector(self.velocity)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position": self.position_vector(),
            "velocity": None if self.velocity is None else self.velocity_vector(),
            "mass": self.mass,
            "radius": self.radius,
            "attributes": self.attributes,
        }


@dataclass
class Trajectory:
    """A rollout as object states over time."""

    states: list[list[ObjectState]]
    actions: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def object_at(self, time_index: int, object_id: str) -> ObjectState | None:
        for obj in self.states[time_index]:
            if obj.object_id == object_id:
                return obj
        return None

    def object_ids(self) -> set[str]:
        return {obj.object_id for frame in self.states for obj in frame}

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "states": [[obj.to_json_dict() for obj in frame] for frame in self.states],
            "actions": self.actions,
            "metadata": self.metadata,
        }


@dataclass
class AuditInput:
    """Inputs consumed by the v0.1 audit pipeline."""

    predicted: Trajectory
    ground_truth: Trajectory | None = None
    observed: Trajectory | None = None
    interventions: list[dict[str, Any]] | None = None
    semantic_graph: dict[str, Any] | None = None
    causal_graph: dict[str, Any] | None = None
    domain_contract: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted.to_json_dict(),
            "ground_truth": None if self.ground_truth is None else self.ground_truth.to_json_dict(),
            "observed": None if self.observed is None else self.observed.to_json_dict(),
            "interventions": self.interventions,
            "semantic_graph": self.semantic_graph,
            "causal_graph": self.causal_graph,
            "domain_contract": self.domain_contract,
        }


@dataclass
class Finding:
    """One audit finding."""

    severity: Severity
    category: Category
    message: str
    time_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "time_index": self.time_index,
            "details": self.details,
        }


@dataclass
class AuditReport:
    """Final v0.1 audit report."""

    overall_risk: Risk
    findings: list[Finding]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "overall_risk": self.overall_risk,
            "findings": [finding.to_json_dict() for finding in self.findings],
            "metrics": self.metrics,
        }

    def to_markdown(self) -> str:
        lines = ["# MechanismLens Audit Report", "", f"Overall risk: **{self.overall_risk}**", ""]
        if self.metrics:
            lines.extend(["## Metrics", ""])
            for name, value in self.metrics.items():
                lines.append(f"- `{name}`: `{json.dumps(value, sort_keys=True)}`")
            lines.append("")
        lines.extend(["## Findings", ""])
        if not self.findings:
            lines.append("No findings.")
        else:
            for finding in self.findings:
                when = "" if finding.time_index is None else f" at t={finding.time_index}"
                lines.append(
                    f"- **{finding.severity}** `{finding.category}`{when}: {finding.message}"
                )
        return "\n".join(lines).rstrip() + "\n"

    def save_markdown(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown(), encoding="utf-8")
