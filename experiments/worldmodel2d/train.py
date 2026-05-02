"""Training utilities for the learned 2D world-model experiment."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from experiments.worldmodel2d.dataset import generate_transition_dataset
from experiments.worldmodel2d.models import LinearDynamicsModel, TorchMLPDynamicsModel


class DynamicsModel(Protocol):
    """Minimal model interface used by evaluation."""

    name: str

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict one next state."""

    def rollout(self, initial_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out a fixed action sequence."""


def train_models(
    n_train: int = 2_000,
    seed: int = 0,
    rare_collision_fraction: float = 0.05,
    train_torch: bool = True,
) -> dict[str, DynamicsModel]:
    """Train the NumPy linear baseline and, if available, a small torch MLP."""

    x_train, y_train = generate_transition_dataset(
        n=n_train,
        seed=seed,
        rare_collision_fraction=rare_collision_fraction,
    )
    models: dict[str, DynamicsModel] = {}

    linear = LinearDynamicsModel()
    linear.fit(x_train, y_train)
    models[linear.name] = linear

    if train_torch:
        try:
            mlp = TorchMLPDynamicsModel(seed=seed)
            mlp.fit(x_train, y_train)
            models[mlp.name] = mlp
        except RuntimeError:
            pass

    return models
