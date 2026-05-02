"""Deterministic 2D toy environment for learned world-model audits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mechanismlens.schema import ObjectState, Trajectory

Policy = Callable[[np.ndarray, int], Sequence[float]]


@dataclass(frozen=True)
class World2DConfig:
    """Configuration for the 2D world-model experiment."""

    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0))
    dt: float = 0.12
    agent_radius: float = 0.08
    obstacle_position: tuple[float, float] = (0.25, 0.0)
    obstacle_radius: float = 0.22
    max_speed: float = 1.6
    action_scale: float = 1.0


class WorldModel2DEnv:
    """A small deterministic 2D dynamics environment with collisions."""

    def __init__(self, config: World2DConfig | None = None) -> None:
        self.config = config or World2DConfig()
        self.state = np.zeros(4, dtype=float)
        self.rng = np.random.default_rng(0)
        self.regime = "id"

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned center-position bounds used by MechanismLens."""

        return self.config.bounds

    def reset(self, seed: int = 0, regime: str = "id") -> np.ndarray:
        """Reset to a deterministic initial state for a regime."""

        self.rng = np.random.default_rng(seed)
        self.regime = regime
        if regime == "id":
            position = np.array([-0.65, -0.55], dtype=float) + self.rng.normal(0.0, 0.05, 2)
            velocity = np.array([0.45, 0.18], dtype=float) + self.rng.normal(0.0, 0.03, 2)
        elif regime == "boundary":
            position = np.array([0.86, 0.62], dtype=float) + self.rng.normal(0.0, 0.03, 2)
            velocity = np.array([0.85, 0.35], dtype=float) + self.rng.normal(0.0, 0.02, 2)
        elif regime == "collision":
            position = np.array([-0.22, 0.02], dtype=float) + self.rng.normal(0.0, 0.025, 2)
            velocity = np.array([0.85, 0.0], dtype=float) + self.rng.normal(0.0, 0.02, 2)
        elif regime == "planner":
            position = np.array([-0.72, 0.0], dtype=float) + self.rng.normal(0.0, 0.02, 2)
            velocity = np.array([0.65, 0.0], dtype=float) + self.rng.normal(0.0, 0.01, 2)
        else:
            raise ValueError(f"Unknown regime: {regime}")
        self.state = np.concatenate([position, velocity]).astype(float)
        self.state = self._resolve_obstacle(self._resolve_bounds(self.state))
        return self.state.copy()

    def step(self, action: Sequence[float]) -> np.ndarray:
        """Advance the environment by one step using ground-truth collision handling."""

        action_vec = np.asarray(action, dtype=float)
        if action_vec.shape != (2,):
            raise ValueError("Action must have shape (2,)")
        action_vec = np.clip(action_vec, -1.0, 1.0) * self.config.action_scale
        position = self.state[:2].copy()
        velocity = self.state[2:].copy()
        velocity = velocity + action_vec * self.config.dt
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_speed:
            velocity = velocity / speed * self.config.max_speed
        position = position + velocity * self.config.dt
        next_state = np.concatenate([position, velocity])
        next_state = self._resolve_obstacle(self._resolve_bounds(next_state))
        self.state = next_state
        return self.state.copy()

    def rollout(self, policy: Policy, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Roll out a policy from the current state."""

        states = [self.state.copy()]
        actions: list[np.ndarray] = []
        for time_index in range(horizon):
            action = np.asarray(policy(self.state.copy(), time_index), dtype=float)
            actions.append(action)
            states.append(self.step(action))
        return np.asarray(states, dtype=float), np.asarray(actions, dtype=float)

    def state_to_trajectory(
        self,
        states: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[Sequence[float]] | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Convert agent states into a MechanismLens trajectory."""

        frames: list[list[ObjectState]] = []
        for state in np.asarray(states, dtype=float):
            frames.append(
                [
                    ObjectState(
                        "agent",
                        label="rigid",
                        position=state[:2].tolist(),
                        velocity=state[2:].tolist(),
                        radius=self.config.agent_radius,
                    ),
                    ObjectState(
                        "obstacle",
                        label="static obstacle",
                        position=list(self.config.obstacle_position),
                        velocity=[0.0, 0.0],
                        radius=self.config.obstacle_radius,
                    ),
                ]
            )
        action_dicts = None
        if actions is not None:
            action_dicts = [
                {"ax": float(action[0]), "ay": float(action[1])}
                for action in np.asarray(actions, dtype=float)
            ]
        return Trajectory(states=frames, actions=action_dicts, metadata=metadata or {})

    def penetration_depth(self, state: Sequence[float] | np.ndarray) -> float:
        """Return agent-obstacle penetration depth for a state."""

        position = np.asarray(state, dtype=float)[:2]
        obstacle = np.asarray(self.config.obstacle_position, dtype=float)
        distance = float(np.linalg.norm(position - obstacle))
        return max(self.config.agent_radius + self.config.obstacle_radius - distance, 0.0)

    def _resolve_bounds(self, state: np.ndarray) -> np.ndarray:
        position = state[:2].copy()
        velocity = state[2:].copy()
        for dim, (lower, upper) in enumerate(self.config.bounds):
            if position[dim] < lower:
                position[dim] = lower
                velocity[dim] = abs(velocity[dim]) * 0.75
            elif position[dim] > upper:
                position[dim] = upper
                velocity[dim] = -abs(velocity[dim]) * 0.75
        return np.concatenate([position, velocity])

    def _resolve_obstacle(self, state: np.ndarray) -> np.ndarray:
        position = state[:2].copy()
        velocity = state[2:].copy()
        obstacle = np.asarray(self.config.obstacle_position, dtype=float)
        delta = position - obstacle
        distance = float(np.linalg.norm(delta))
        min_distance = self.config.agent_radius + self.config.obstacle_radius
        if distance >= min_distance:
            return state
        normal = np.array([-1.0, 0.0], dtype=float) if distance < 1e-12 else delta / distance
        position = obstacle + normal * min_distance
        normal_velocity = float(np.dot(velocity, normal))
        if normal_velocity < 0.0:
            velocity = velocity - 1.6 * normal_velocity * normal
        return np.concatenate([position, velocity])


def scripted_policy(regime: str) -> Policy:
    """Return a deterministic policy for evaluation rollouts."""

    def policy(state: np.ndarray, time_index: int) -> np.ndarray:
        if regime == "boundary":
            return np.array([0.35, 0.05], dtype=float)
        if regime == "collision":
            return np.array([0.8, -0.03], dtype=float)
        if regime == "planner":
            target = np.array([0.82, 0.0], dtype=float)
            direction = target - state[:2]
            norm = max(float(np.linalg.norm(direction)), 1e-9)
            return np.clip(direction / norm * 0.9, -1.0, 1.0)
        return np.array([0.08, -0.02], dtype=float)

    return policy
