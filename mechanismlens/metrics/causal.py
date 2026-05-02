"""Causal validity metrics."""

from __future__ import annotations

from typing import Any

from mechanismlens.schema import Trajectory, vector_distance


def intervention_locality_score(
    predicted_base: Trajectory,
    predicted_intervened: Trajectory,
    affected_object_ids: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Estimate whether intervention effects concentrate on expected affected objects."""

    affected = set(affected_object_ids)
    affected_delta = 0.0
    unaffected_delta = 0.0
    affected_count = 0
    unaffected_count = 0

    horizon = min(len(predicted_base.states), len(predicted_intervened.states))
    for time_index in range(horizon):
        base_by_id = {obj.object_id: obj for obj in predicted_base.states[time_index]}
        intervened_by_id = {obj.object_id: obj for obj in predicted_intervened.states[time_index]}
        for object_id in set(base_by_id).intersection(intervened_by_id):
            delta = vector_distance(base_by_id[object_id].position, intervened_by_id[object_id].position)
            if object_id in affected:
                affected_delta += delta
                affected_count += 1
            else:
                unaffected_delta += delta
                unaffected_count += 1

    affected_mean = affected_delta / affected_count if affected_count else 0.0
    unaffected_mean = unaffected_delta / unaffected_count if unaffected_count else 0.0
    total = affected_mean + unaffected_mean
    locality = 1.0 if total == 0.0 else affected_mean / total
    return {
        "locality_score": locality,
        "affected_mean_delta": affected_mean,
        "unaffected_mean_delta": unaffected_mean,
    }
