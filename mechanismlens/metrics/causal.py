"""Counterfactual causal audit metrics."""

from __future__ import annotations

from typing import Any

from mechanismlens.schema import Finding, Trajectory, vector_distance


def per_object_deviation(base: Trajectory, intervened: Trajectory) -> dict[str, float]:
    """Compute mean position deviation per matched object over matched timesteps."""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    horizon = min(len(base.states), len(intervened.states))

    for time_index in range(horizon):
        base_by_id = {obj.object_id: obj for obj in base.states[time_index]}
        intervened_by_id = {obj.object_id: obj for obj in intervened.states[time_index]}
        for object_id in sorted(set(base_by_id).intersection(intervened_by_id)):
            delta = vector_distance(base_by_id[object_id].position, intervened_by_id[object_id].position)
            totals[object_id] = totals.get(object_id, 0.0) + delta
            counts[object_id] = counts.get(object_id, 0) + 1

    return {object_id: totals[object_id] / counts[object_id] for object_id in sorted(totals)}


def intervention_locality_score(
    base: Trajectory,
    intervened: Trajectory,
    expected_affected_object_ids: set[str] | list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Estimate whether intervention effects concentrate on expected affected objects."""

    expected = set(expected_affected_object_ids or [])
    deviations = per_object_deviation(base, intervened)
    affected_values = [delta for object_id, delta in deviations.items() if object_id in expected]
    unaffected_values = [delta for object_id, delta in deviations.items() if object_id not in expected]

    affected_mean = sum(affected_values) / len(affected_values) if affected_values else 0.0
    unaffected_mean = sum(unaffected_values) / len(unaffected_values) if unaffected_values else 0.0
    total = affected_mean + unaffected_mean
    locality = 1.0 if total == 0.0 else affected_mean / total

    return {
        "locality_score": locality,
        "affected_mean_delta": affected_mean,
        "unaffected_mean_delta": unaffected_mean,
        "per_object_deviation": deviations,
    }


def unexpected_side_effect_findings(
    base: Trajectory,
    intervened: Trajectory,
    expected_affected_object_ids: set[str] | list[str] | tuple[str, ...] | None,
    threshold: float = 0.1,
) -> list[Finding]:
    """Emit causal findings for unexpected object changes after an intervention."""

    expected = set(expected_affected_object_ids or [])
    findings: list[Finding] = []
    for object_id, delta in per_object_deviation(base, intervened).items():
        if object_id in expected or delta <= threshold:
            continue
        severity = "high" if delta > 0.5 else "medium"
        findings.append(
            Finding(
                severity=severity,
                category="causal",
                message=(
                    f"Unexpected side effect: object {object_id} changes by {delta:.3f} "
                    "despite not being in the expected affected set."
                ),
                details={
                    "object_id": object_id,
                    "mean_position_deviation": delta,
                    "threshold": threshold,
                    "expected_affected_object_ids": sorted(expected),
                },
            )
        )
    return findings
