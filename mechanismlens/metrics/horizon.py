"""Prediction-horizon metrics."""

from __future__ import annotations

from typing import Any

from mechanismlens.schema import Trajectory, vector_distance


def mean_position_error(
    predicted: Trajectory | None,
    ground_truth: Trajectory | None,
) -> list[float]:
    """Return mean object-position error per timestep.

    Missing ground truth, missing predictions, or timesteps with no shared object ids return
    an empty list or skip that timestep gracefully.
    """

    if predicted is None or ground_truth is None:
        return []

    errors: list[float] = []
    horizon = min(len(predicted.states), len(ground_truth.states))
    for time_index in range(horizon):
        pred_by_id = {obj.object_id: obj for obj in predicted.states[time_index]}
        truth_by_id = {obj.object_id: obj for obj in ground_truth.states[time_index]}
        shared_ids = sorted(set(pred_by_id).intersection(truth_by_id))
        if not shared_ids:
            continue
        distances = [
            vector_distance(pred_by_id[object_id].position, truth_by_id[object_id].position)
            for object_id in shared_ids
        ]
        errors.append(sum(distances) / len(distances))
    return errors


def horizon_amplification(errors: list[float]) -> dict[str, Any]:
    """Summarize whether rollout error amplifies over the horizon."""

    if not errors:
        return {"h1": None, "hfinal": None, "h1_hfinal_ratio": None, "hfinal_h1_ratio": None}

    h1 = errors[0]
    hfinal = errors[-1]
    return {
        "h1": h1,
        "hfinal": hfinal,
        "h1_hfinal_ratio": None if hfinal == 0 else h1 / hfinal,
        "hfinal_h1_ratio": None if h1 == 0 else hfinal / h1,
    }
