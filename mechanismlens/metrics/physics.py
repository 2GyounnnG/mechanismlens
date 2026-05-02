"""Toy rigid-body feasibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechanismlens.schema import Finding, ObjectState, Trajectory, vector_distance


def _bounds_for_dimension(bounds: Any, dimension: int) -> tuple[float, float] | None:
    if bounds is None:
        return None
    if isinstance(bounds, dict):
        key = ("x", "y", "z")[dimension] if dimension < 3 else str(dimension)
        value = bounds.get(key, bounds.get(dimension))
        return None if value is None else (float(value[0]), float(value[1]))
    if dimension >= len(bounds):
        return None
    value = bounds[dimension]
    return (float(value[0]), float(value[1]))


def boundary_violation(
    trajectory: Trajectory,
    bounds: Sequence[tuple[float, float]] | dict[str | int, tuple[float, float]] | None,
) -> tuple[float, list[Finding]]:
    """Detect objects outside axis-aligned bounds.

    Returns max violation distance and findings.
    """

    if bounds is None:
        return 0.0, []

    max_violation = 0.0
    findings: list[Finding] = []
    for time_index, frame in enumerate(trajectory.states):
        for obj in frame:
            for dimension, value in enumerate(obj.position_vector()):
                span = _bounds_for_dimension(bounds, dimension)
                if span is None:
                    continue
                lower, upper = span
                violation = max(lower - value, value - upper, 0.0)
                if violation > 0.0:
                    max_violation = max(max_violation, violation)
                    findings.append(
                        Finding(
                            severity="high",
                            category="physics",
                            message=f"Object {obj.object_id} is outside boundary on dimension {dimension}.",
                            time_index=time_index,
                            details={
                                "object_id": obj.object_id,
                                "dimension": dimension,
                                "value": value,
                                "bounds": [lower, upper],
                                "violation": violation,
                            },
                        )
                    )
    return max_violation, findings


def penetration_violation(trajectory: Trajectory) -> tuple[float, list[Finding]]:
    """Detect overlap between circular or spherical objects with radii."""

    max_penetration = 0.0
    findings: list[Finding] = []
    for time_index, frame in enumerate(trajectory.states):
        for left_index, left in enumerate(frame):
            if left.radius is None:
                continue
            for right in frame[left_index + 1 :]:
                if right.radius is None:
                    continue
                distance = vector_distance(left.position, right.position)
                penetration = left.radius + right.radius - distance
                if penetration > 0.0:
                    max_penetration = max(max_penetration, penetration)
                    findings.append(
                        Finding(
                            severity="high",
                            category="physics",
                            message=(
                                f"Objects {left.object_id} and {right.object_id} penetrate by "
                                f"{penetration:.3f}."
                            ),
                            time_index=time_index,
                            details={
                                "object_ids": [left.object_id, right.object_id],
                                "distance": distance,
                                "penetration": penetration,
                            },
                        )
                    )
    return max_penetration, findings


def _momentum(frame: list[ObjectState]) -> list[float] | None:
    total: list[float] | None = None
    for obj in frame:
        if obj.velocity is None or obj.mass is None:
            continue
        contribution = [obj.mass * value for value in obj.velocity_vector()]
        if total is None:
            total = [0.0 for _ in contribution]
        if len(contribution) != len(total):
            continue
        total = [current + delta for current, delta in zip(total, contribution, strict=True)]
    return total


def momentum_drift(trajectory: Trajectory) -> tuple[float, list[Finding]]:
    """Estimate total momentum drift relative to the first measurable timestep."""

    momenta = [_momentum(frame) for frame in trajectory.states]
    indexed = [(idx, value) for idx, value in enumerate(momenta) if value is not None]
    if len(indexed) < 2:
        return 0.0, []

    baseline_index, baseline = indexed[0]
    baseline_norm = max(sum(value * value for value in baseline) ** 0.5, 1e-12)
    max_drift = 0.0
    worst_time = baseline_index
    for time_index, current in indexed[1:]:
        if len(current) != len(baseline):
            continue
        drift = (
            sum((cur - base) ** 2 for cur, base in zip(current, baseline, strict=True)) ** 0.5
            / baseline_norm
        )
        if drift > max_drift:
            max_drift = drift
            worst_time = time_index

    findings: list[Finding] = []
    if max_drift > 0.1:
        findings.append(
            Finding(
                severity="medium",
                category="physics",
                message=f"Total momentum drifts by {max_drift:.3f} relative to the first timestep.",
                time_index=worst_time,
                details={"momentum_drift": max_drift},
            )
        )
    return max_drift, findings
