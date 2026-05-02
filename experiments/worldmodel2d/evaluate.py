"""Evaluation for learned 2D world-model rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from experiments.worldmodel2d.dataset import Regime, generate_rollout_initial_conditions
from experiments.worldmodel2d.env import WorldModel2DEnv, scripted_policy
from experiments.worldmodel2d.train import DynamicsModel
from mechanismlens import AuditInput, AuditSuite
from mechanismlens.experiments.analysis import severity_weighted_risk_score
from mechanismlens.schema import CATEGORY_ORDER

DEFAULT_REGIMES: tuple[Regime, ...] = ("id", "boundary", "collision", "planner")


def evaluate_models(
    models: Mapping[str, DynamicsModel],
    n_rollouts: int = 20,
    horizon: int = 16,
    seed: int = 1_000,
    regimes: tuple[Regime, ...] = DEFAULT_REGIMES,
) -> list[dict[str, Any]]:
    """Evaluate trained models on rollouts and MechanismLens audit metrics."""

    env = WorldModel2DEnv()
    suite = AuditSuite(bounds=env.bounds)
    records: list[dict[str, Any]] = []

    for model_name, model in models.items():
        for regime_index, regime in enumerate(regimes):
            initials = generate_rollout_initial_conditions(
                n=n_rollouts,
                regime=regime,
                seed=seed + regime_index * 10_000,
            )
            for rollout_id, initial_state in enumerate(initials):
                record = _evaluate_one(
                    model_name=model_name,
                    model=model,
                    initial_state=initial_state,
                    regime=regime,
                    rollout_id=rollout_id,
                    horizon=horizon,
                    suite=suite,
                )
                records.append(record)
    return records


def _evaluate_one(
    model_name: str,
    model: DynamicsModel,
    initial_state: np.ndarray,
    regime: Regime,
    rollout_id: int,
    horizon: int,
    suite: AuditSuite,
) -> dict[str, Any]:
    env = WorldModel2DEnv()
    env.state = np.asarray(initial_state, dtype=float).copy()
    policy = scripted_policy(regime)
    ground_truth_states, actions = env.rollout(policy, horizon)
    predicted_states = model.rollout(initial_state, actions)
    one_step_predictions = np.asarray(
        [
            model.predict_next(ground_truth_states[index], actions[index])
            for index in range(horizon)
        ],
        dtype=float,
    )
    one_step_mse = float(np.mean((one_step_predictions - ground_truth_states[1:]) ** 2))
    state_errors = np.mean((predicted_states - ground_truth_states) ** 2, axis=1)
    mse_mean = float(np.mean(state_errors[1:])) if len(state_errors) > 1 else 0.0
    mse_final = float(state_errors[-1]) if len(state_errors) else 0.0

    predicted_trajectory = env.state_to_trajectory(
        predicted_states,
        actions,
        metadata={"model_name": model_name, "regime": regime, "rollout_id": rollout_id},
    )
    ground_truth_trajectory = env.state_to_trajectory(
        ground_truth_states,
        actions,
        metadata={"source": "ground_truth", "regime": regime, "rollout_id": rollout_id},
    )

    predicted_rewards: list[float] | None = None
    realized_rewards: list[float] | None = None
    if regime == "planner":
        predicted_rewards = _planner_rewards(env, predicted_states, optimistic=True)
        realized_rewards = _planner_rewards(env, ground_truth_states, optimistic=False)

    report = suite.run(
        AuditInput(
            predicted=predicted_trajectory,
            ground_truth=ground_truth_trajectory,
            predicted_rewards=predicted_rewards,
            realized_rewards=realized_rewards,
            planner_metadata={"model_name": model_name, "regime": regime}
            if regime == "planner"
            else None,
        )
    )
    category_counts = report.category_counts()
    decision_gap = report.metrics.get("decision_return_gap", {})
    return_gap = decision_gap.get("return_gap") if isinstance(decision_gap, dict) else None

    record: dict[str, Any] = {
        "model_name": model_name,
        "regime": regime,
        "rollout_id": rollout_id,
        "one_step_mse": one_step_mse,
        "mse_mean": mse_mean,
        "mse_final": mse_final,
        "audit_risk_score": severity_weighted_risk_score(report),
        "overall_risk": report.overall_risk,
        "finding_count": len(report.findings),
        "return_gap": float(return_gap) if isinstance(return_gap, (int, float)) else None,
    }
    for category in CATEGORY_ORDER:
        record[f"{category}_count"] = category_counts.get(category, 0)
    return record


def _planner_rewards(
    env: WorldModel2DEnv,
    states: np.ndarray,
    *,
    optimistic: bool,
) -> list[float]:
    target = np.array([0.82, 0.0], dtype=float)
    rewards: list[float] = []
    for state in states[1:]:
        distance = float(np.linalg.norm(state[:2] - target))
        reward = 1.0 - distance
        if state[0] > 0.72:
            reward += 0.8
        if not optimistic:
            obstacle = np.array(env.config.obstacle_position, dtype=float)
            contact_distance = env.config.agent_radius + env.config.obstacle_radius + 0.01
            if float(np.linalg.norm(state[:2] - obstacle)) <= contact_distance:
                reward -= 1.5
        rewards.append(reward)
    return rewards
