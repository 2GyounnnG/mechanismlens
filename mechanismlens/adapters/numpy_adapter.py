"""Array/list adapters for building trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechanismlens.schema import ObjectState, Trajectory


def trace_from_arrays(
    positions: Mapping[str, Sequence[Any]],
    *,
    labels: Mapping[str, str] | None = None,
    velocities: Mapping[str, Sequence[Any]] | None = None,
    masses: Mapping[str, float] | None = None,
    radii: Mapping[str, float] | None = None,
    name: str = "array_trajectory",
) -> Trajectory:
    """Build a trajectory from per-object position arrays or lists."""

    lengths = {len(value) for value in positions.values()}
    if not lengths:
        raise ValueError("at least one object position series is required")
    if len(lengths) != 1:
        raise ValueError("all position series must have the same length")

    horizon = lengths.pop()
    frames: list[list[ObjectState]] = []
    for time_index in range(horizon):
        frame = []
        for object_id, series in positions.items():
            frame.append(
                ObjectState(
                    object_id=object_id,
                    label=None if labels is None else labels.get(object_id),
                    position=_to_list(series[time_index]),
                    velocity=None
                    if velocities is None or object_id not in velocities
                    else _to_list(velocities[object_id][time_index]),
                    mass=None if masses is None else masses.get(object_id),
                    radius=None if radii is None else radii.get(object_id),
                )
            )
        frames.append(frame)
    return Trajectory(states=frames, metadata={"name": name})


def _to_list(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value
