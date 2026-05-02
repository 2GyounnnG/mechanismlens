"""Dataset generation for the 2D learned world-model experiment."""

from __future__ import annotations

from typing import Literal

import numpy as np

from experiments.worldmodel2d.env import WorldModel2DEnv

Regime = Literal["id", "boundary", "collision", "planner"]


def generate_transition_dataset(
    n: int,
    seed: int,
    rare_collision_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate supervised transition pairs `(state, action) -> next_state`."""

    rng = np.random.default_rng(seed)
    env = WorldModel2DEnv()
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    collision_every = max(int(round(1.0 / max(rare_collision_fraction, 1e-9))), 1)
    for index in range(n):
        regime = (
            "collision"
            if rare_collision_fraction > 0 and index % collision_every == 0
            else "id"
        )
        state = env.reset(seed=seed * 10_000 + index, regime=regime)
        if regime == "id":
            action = rng.uniform(-0.35, 0.35, size=2)
        else:
            action = np.array([0.8, rng.uniform(-0.08, 0.08)], dtype=float)
        next_state = env.step(action)
        x_rows.append(np.concatenate([state, action]))
        y_rows.append(next_state)
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def generate_rollout_initial_conditions(n: int, regime: Regime, seed: int) -> list[np.ndarray]:
    """Generate deterministic initial states for evaluation rollouts."""

    env = WorldModel2DEnv()
    return [env.reset(seed=seed * 10_000 + index, regime=regime).copy() for index in range(n)]
