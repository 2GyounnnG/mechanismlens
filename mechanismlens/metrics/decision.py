"""Decision-risk metrics for planning traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechanismlens.schema import Finding


def action_switch_rate(actions: Sequence[str]) -> float:
    """Return the fraction of adjacent action pairs that differ."""

    if len(actions) < 2:
        return 0.0
    switches = sum(1 for prev, cur in zip(actions, actions[1:], strict=False) if prev != cur)
    return switches / (len(actions) - 1)


def imagined_real_return_gap(
    predicted_rewards: Sequence[float] | None,
    realized_rewards: Sequence[float] | None,
) -> dict[str, float]:
    """Compare imagined return against realized return."""

    predicted_return = sum(float(reward) for reward in predicted_rewards or [])
    realized_return = sum(float(reward) for reward in realized_rewards or [])
    return {
        "predicted_return": predicted_return,
        "realized_return": realized_return,
        "return_gap": predicted_return - realized_return,
    }


def uncertainty_on_planned_path(uncertainty: Sequence[float] | None) -> dict[str, float]:
    """Summarize uncertainty along the planned path."""

    values = [float(value) for value in uncertainty or []]
    if not values:
        return {"mean_uncertainty": 0.0, "max_uncertainty": 0.0}
    return {
        "mean_uncertainty": sum(values) / len(values),
        "max_uncertainty": max(values),
    }


def violation_reward_coupling(
    predicted_rewards: Sequence[float] | None,
    violation_by_timestep: Sequence[bool] | Mapping[int, bool] | None,
) -> dict[str, Any]:
    """Compare predicted reward on violation and non-violation timesteps."""

    rewards = [float(reward) for reward in predicted_rewards or []]
    if isinstance(violation_by_timestep, Mapping):
        violations = [bool(violation_by_timestep.get(idx, False)) for idx in range(len(rewards))]
    else:
        raw = list(violation_by_timestep or [])
        violations = [bool(raw[idx]) if idx < len(raw) else False for idx in range(len(rewards))]

    violation_rewards = [reward for reward, violated in zip(rewards, violations, strict=False) if violated]
    non_violation_rewards = [
        reward for reward, violated in zip(rewards, violations, strict=False) if not violated
    ]
    violation_mean = (
        sum(violation_rewards) / len(violation_rewards) if violation_rewards else 0.0
    )
    non_violation_mean = (
        sum(non_violation_rewards) / len(non_violation_rewards) if non_violation_rewards else 0.0
    )
    return {
        "violation_reward_mean": violation_mean,
        "non_violation_reward_mean": non_violation_mean,
        "violation_reward_gap": violation_mean - non_violation_mean,
        "violation_timesteps": [idx for idx, violated in enumerate(violations) if violated],
    }


def decision_risk_findings(
    predicted_rewards: Sequence[float] | None = None,
    realized_rewards: Sequence[float] | None = None,
    uncertainty: Sequence[float] | None = None,
    violation_by_timestep: Sequence[bool] | Mapping[int, bool] | None = None,
    *,
    return_gap_medium_threshold: float = 0.25,
    return_gap_high_threshold: float = 1.0,
    mean_uncertainty_threshold: float = 0.5,
    max_uncertainty_threshold: float = 0.8,
    high_reward_threshold: float = 0.75,
) -> list[Finding]:
    """Generate decision-risk findings from reward, uncertainty, and violation signals."""

    findings: list[Finding] = []
    gap_metrics = imagined_real_return_gap(predicted_rewards, realized_rewards)
    return_gap = gap_metrics["return_gap"]
    if return_gap > return_gap_high_threshold:
        severity = "high"
    elif return_gap > return_gap_medium_threshold:
        severity = "medium"
    else:
        severity = None
    if severity is not None:
        findings.append(
            Finding(
                severity=severity,
                category="decision",
                message=(
                    f"Planner imagined return exceeds realized return by {return_gap:.3f}."
                ),
                details=gap_metrics,
            )
        )

    coupling = violation_reward_coupling(predicted_rewards, violation_by_timestep)
    if coupling["violation_timesteps"] and coupling["violation_reward_mean"] > high_reward_threshold:
        reward_gap = coupling["violation_reward_gap"]
        severity = "high" if reward_gap > 1.0 else "medium"
        findings.append(
            Finding(
                severity=severity,
                category="decision",
                message=(
                    "High predicted reward coincides with physics/causal/cross-layer "
                    "violation timesteps."
                ),
                details=coupling,
            )
        )

    uncertainty_metrics = uncertainty_on_planned_path(uncertainty)
    mean_uncertainty = uncertainty_metrics["mean_uncertainty"]
    max_uncertainty = uncertainty_metrics["max_uncertainty"]
    if mean_uncertainty > mean_uncertainty_threshold or max_uncertainty > max_uncertainty_threshold:
        severity = (
            "high"
            if mean_uncertainty > mean_uncertainty_threshold * 1.6
            or max_uncertainty > max_uncertainty_threshold * 1.5
            else "medium"
        )
        findings.append(
            Finding(
                severity=severity,
                category="decision",
                message=(
                    "Planner path uncertainty exceeds configured decision-risk thresholds."
                ),
                details={
                    **uncertainty_metrics,
                    "mean_uncertainty_threshold": mean_uncertainty_threshold,
                    "max_uncertainty_threshold": max_uncertainty_threshold,
                },
            )
        )

    return findings
