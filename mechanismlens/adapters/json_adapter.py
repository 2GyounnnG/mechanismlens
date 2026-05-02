"""JSON adapters for trajectories and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mechanismlens.schema import ObjectState, Trajectory


def object_state_from_dict(payload: dict[str, Any]) -> ObjectState:
    return ObjectState(
        object_id=payload["object_id"],
        label=payload.get("label"),
        position=payload.get("position", []),
        velocity=payload.get("velocity"),
        mass=payload.get("mass"),
        radius=payload.get("radius"),
        attributes=payload.get("attributes", {}),
    )


def trajectory_from_json(data: str | bytes | dict[str, Any]) -> Trajectory:
    payload = json.loads(data) if isinstance(data, (str, bytes)) else data
    return Trajectory(
        states=[
            [object_state_from_dict(obj_payload) for obj_payload in frame]
            for frame in payload.get("states", [])
        ],
        actions=payload.get("actions"),
        metadata=payload.get("metadata", {}),
    )


def load_trajectory(path: str | Path) -> Trajectory:
    return trajectory_from_json(Path(path).read_text(encoding="utf-8"))


def dump_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(trajectory.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
