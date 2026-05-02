"""Lightweight dynamics models for the 2D world-model experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class LinearDynamicsModel:
    """Ridge-regression dynamics model using NumPy."""

    name = "linear"

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit `(state, action) -> next_state` using ridge regression."""

        design = _with_bias(np.asarray(x, dtype=float))
        targets = np.asarray(y, dtype=float)
        penalty = self.ridge * np.eye(design.shape[1], dtype=float)
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

    def predict_next(self, state: Sequence[float], action: Sequence[float]) -> np.ndarray:
        """Predict one next state."""

        if self.weights is None:
            raise RuntimeError("LinearDynamicsModel must be fitted before prediction")
        row = _with_bias(np.asarray([np.concatenate([state, action])], dtype=float))
        return np.asarray(row @ self.weights, dtype=float)[0]

    def rollout(
        self,
        initial_state: Sequence[float],
        actions: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Roll out the learned model for a fixed action sequence."""

        states = [np.asarray(initial_state, dtype=float)]
        current = states[0]
        for action in actions:
            current = self.predict_next(current, action)
            states.append(current)
        return np.asarray(states, dtype=float)


class TorchMLPDynamicsModel:
    """Small optional PyTorch MLP dynamics model."""

    name = "torch_mlp"

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 80,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not available") from exc
        self.torch = torch
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model: Any | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the MLP if torch is installed."""

        torch = self.torch
        torch.manual_seed(self.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 4),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        inputs = torch.as_tensor(np.asarray(x, dtype=np.float32))
        targets = torch.as_tensor(np.asarray(y, dtype=np.float32))
        for _ in range(self.epochs):
            optimizer.zero_grad()
            prediction = self.model(inputs)
            loss = torch.mean((prediction - targets) ** 2)
            loss.backward()
            optimizer.step()

    def predict_next(self, state: Sequence[float], action: Sequence[float]) -> np.ndarray:
        """Predict one next state."""

        if self.model is None:
            raise RuntimeError("TorchMLPDynamicsModel must be fitted before prediction")
        torch = self.torch
        row = np.asarray([np.concatenate([state, action])], dtype=np.float32)
        with torch.no_grad():
            prediction = self.model(torch.as_tensor(row)).detach().cpu().numpy()[0]
        return np.asarray(prediction, dtype=float)

    def rollout(
        self,
        initial_state: Sequence[float],
        actions: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Roll out the learned model for a fixed action sequence."""

        states = [np.asarray(initial_state, dtype=float)]
        current = states[0]
        for action in actions:
            current = self.predict_next(current, action)
            states.append(current)
        return np.asarray(states, dtype=float)


def torch_available() -> bool:
    """Return whether PyTorch can be imported."""

    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _with_bias(x: np.ndarray) -> np.ndarray:
    bias = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([x, bias], axis=1)
