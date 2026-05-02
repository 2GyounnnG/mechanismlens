"""Cross-layer semantic and physical consistency checks."""

from __future__ import annotations

from mechanismlens.schema import Finding, ObjectState, Trajectory, vector_distance


def semantic_physics_mismatch(trajectory: Trajectory) -> list[Finding]:
    """Find contradictions between object labels and physical behavior."""

    findings: list[Finding] = []
    history: dict[str, list[tuple[int, ObjectState]]] = {}
    for time_index, frame in enumerate(trajectory.states):
        for obj in frame:
            history.setdefault(obj.object_id, []).append((time_index, obj))
            label = (obj.label or "").lower()
            if "obstacle" in label and obj.radius is None:
                findings.append(
                    Finding(
                        severity="medium",
                        category="cross_layer",
                        message=f"Obstacle {obj.object_id} is missing radius/physical support.",
                        time_index=time_index,
                        details={"object_id": obj.object_id, "label": obj.label},
                    )
                )

    for object_id, entries in history.items():
        first_time, first = entries[0]
        first_label = (first.label or "").lower()
        first_position = first.position
        first_radius = first.radius
        for time_index, obj in entries[1:]:
            label = (obj.label or first_label).lower()
            if "rigid" in label and first_radius is not None and obj.radius is not None:
                if abs(obj.radius - first_radius) > 1e-6:
                    findings.append(
                        Finding(
                            severity="medium",
                            category="cross_layer",
                            message=f"Rigid object {object_id} changes radius over time.",
                            time_index=time_index,
                            details={
                                "object_id": object_id,
                                "initial_time": first_time,
                                "initial_radius": first_radius,
                                "radius": obj.radius,
                            },
                        )
                    )
            if "static" in label:
                displacement = vector_distance(first_position, obj.position)
                if displacement > 1e-3:
                    findings.append(
                        Finding(
                            severity="medium",
                            category="cross_layer",
                            message=f"Static object {object_id} moves by {displacement:.3f}.",
                            time_index=time_index,
                            details={"object_id": object_id, "displacement": displacement},
                        )
                    )
    return findings
