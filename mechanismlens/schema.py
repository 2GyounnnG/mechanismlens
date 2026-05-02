"""Serializable audit schema for MechanismLens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

Severity = Literal["low", "medium", "high"]
Category = Literal["semantic", "causal", "physics", "cross_layer", "decision", "horizon"]
Risk = Literal["low", "medium", "high"]
CATEGORY_ORDER: tuple[Category, ...] = (
    "semantic",
    "causal",
    "physics",
    "cross_layer",
    "decision",
    "horizon",
)
SEVERITY_ORDER: tuple[Severity, ...] = ("low", "medium", "high")


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
    """Inputs consumed by the audit pipeline."""

    predicted: Trajectory
    ground_truth: Trajectory | None = None
    observed: Trajectory | None = None
    interventions: list[dict[str, Any]] | None = None
    semantic_graph: dict[str, Any] | None = None
    causal_graph: dict[str, Any] | None = None
    domain_contract: dict[str, Any] | None = None
    predicted_base: Trajectory | None = None
    predicted_intervened: Trajectory | None = None
    expected_affected_object_ids: list[str] | None = None
    intervention_description: str | None = None
    planned_actions: list[dict[str, Any]] | None = None
    predicted_rewards: list[float] | None = None
    realized_rewards: list[float] | None = None
    uncertainty: list[float] | None = None
    planner_metadata: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted.to_json_dict(),
            "ground_truth": None if self.ground_truth is None else self.ground_truth.to_json_dict(),
            "observed": None if self.observed is None else self.observed.to_json_dict(),
            "interventions": self.interventions,
            "semantic_graph": self.semantic_graph,
            "causal_graph": self.causal_graph,
            "domain_contract": self.domain_contract,
            "predicted_base": None
            if self.predicted_base is None
            else self.predicted_base.to_json_dict(),
            "predicted_intervened": None
            if self.predicted_intervened is None
            else self.predicted_intervened.to_json_dict(),
            "expected_affected_object_ids": self.expected_affected_object_ids,
            "intervention_description": self.intervention_description,
            "planned_actions": self.planned_actions,
            "predicted_rewards": self.predicted_rewards,
            "realized_rewards": self.realized_rewards,
            "uncertainty": self.uncertainty,
            "planner_metadata": self.planner_metadata,
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
    """Final audit report."""

    overall_risk: Risk
    findings: list[Finding]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly report dictionary."""

        return {
            "overall_risk": self.overall_risk,
            "summary": {
                "category_counts": self.category_counts(),
                "severity_counts": self.severity_counts(),
            },
            "findings": [finding.to_json_dict() for finding in self.findings],
            "metrics": self.metrics,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Return the report as JSON and optionally write it to a path."""

        payload = json.dumps(self.to_json_dict(), indent=2, sort_keys=True)
        if path is not None:
            self.save_json(path)
        return payload

    def save_json(self, path: str | Path) -> None:
        """Save the report as deterministic JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")

    def category_counts(self) -> dict[str, int]:
        """Count findings by category in a stable order."""

        counts = {category: 0 for category in CATEGORY_ORDER}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return {category: count for category, count in counts.items() if count > 0}

    def severity_counts(self) -> dict[str, int]:
        """Count findings by severity in a stable order."""

        counts = {severity: 0 for severity in SEVERITY_ORDER}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return {severity: count for severity, count in counts.items() if count > 0}

    def risk_summary_table_markdown(self) -> str:
        """Render a compact category/severity summary table."""

        category_counts = self.category_counts()
        severity_counts = self.severity_counts()
        lines = [
            "| Type | Low | Medium | High | Total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for category in CATEGORY_ORDER:
            category_findings = [finding for finding in self.findings if finding.category == category]
            if not category_findings:
                continue
            low = sum(1 for finding in category_findings if finding.severity == "low")
            medium = sum(1 for finding in category_findings if finding.severity == "medium")
            high = sum(1 for finding in category_findings if finding.severity == "high")
            lines.append(f"| {category} | {low} | {medium} | {high} | {category_counts[category]} |")
        if not category_counts:
            lines.append("| none | 0 | 0 | 0 | 0 |")
        lines.append(
            "| **Total** | "
            f"{severity_counts.get('low', 0)} | "
            f"{severity_counts.get('medium', 0)} | "
            f"{severity_counts.get('high', 0)} | "
            f"{len(self.findings)} |"
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render a deterministic Markdown audit report."""

        from mechanismlens.recommendations import generate_recommendations

        lines = [
            "# MechanismLens Audit Report",
            "",
            f"Overall risk: **{self.overall_risk}**",
            "",
            "## Summary",
            "",
            self.risk_summary_table_markdown(),
            "",
        ]
        if self.metrics:
            lines.extend(["## Metrics", ""])
            for name, value in self.metrics.items():
                lines.append(f"- `{name}`: `{json.dumps(value, sort_keys=True)}`")
            lines.append("")
        lines.extend(["## Findings", ""])
        if not self.findings:
            lines.append("No findings.")
        else:
            for category in CATEGORY_ORDER:
                category_findings = [
                    finding for finding in self.findings if finding.category == category
                ]
                if not category_findings:
                    continue
                lines.extend([f"### {category}", ""])
                for finding in category_findings:
                    when = "" if finding.time_index is None else f" at t={finding.time_index}"
                    lines.append(f"- **{finding.severity}**{when}: {finding.message}")
                lines.append("")
            recommendations = generate_recommendations(self)
            if recommendations:
                lines.extend(["## Recommendations", ""])
                for recommendation in recommendations:
                    lines.append(f"- {recommendation}")
        return "\n".join(lines).rstrip() + "\n"

    def save_markdown(self, path: str | Path) -> None:
        """Save the report as Markdown, creating parent directories as needed."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
