"""MechanismLens audit pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from .contracts.base import DomainContract
from .contracts.toy_rigid_body import ToyRigidBodyContract
from .metrics.causal import intervention_locality_score, unexpected_side_effect_findings
from .metrics.decision import (
    decision_risk_findings,
    imagined_real_return_gap,
    uncertainty_on_planned_path,
    violation_reward_coupling,
)
from .metrics.horizon import horizon_amplification, mean_position_error
from .schema import AuditInput, AuditReport, Finding, Risk

RISK_ORDER: dict[Risk, int] = {"low": 0, "medium": 1, "high": 2}


def _overall_risk(findings: list[Finding]) -> Risk:
    if not findings:
        return "low"
    return max((finding.severity for finding in findings), key=lambda item: RISK_ORDER[item])


def _violation_timesteps(findings: list[Finding], horizon: int) -> list[bool]:
    violation_categories = {"physics", "causal", "cross_layer"}
    violation_by_timestep = [False for _ in range(horizon)]
    for finding in findings:
        if finding.category not in violation_categories or finding.time_index is None:
            continue
        if 0 <= finding.time_index < horizon:
            violation_by_timestep[finding.time_index] = True
    return violation_by_timestep


class AuditSuite:
    """Run rollout and counterfactual checks over an :class:`AuditInput`."""

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]] | dict[str | int, tuple[float, float]] | None = None,
        contract: DomainContract | None = None,
    ) -> None:
        self.bounds = bounds
        self.contract = contract or ToyRigidBodyContract(bounds=bounds)

    def run(self, audit_input: AuditInput) -> AuditReport:
        """Run horizon, physics, cross-layer, and optional counterfactual checks."""

        findings: list[Finding] = []
        metrics: dict[str, Any] = {}

        if audit_input.ground_truth is not None:
            errors = mean_position_error(audit_input.predicted, audit_input.ground_truth)
            metrics["mean_position_error"] = errors
            metrics["horizon_amplification"] = horizon_amplification(errors)

        contract_metrics, contract_findings = self.contract.check_trajectory(audit_input.predicted)
        metrics.update(contract_metrics)
        findings.extend(contract_findings)

        if audit_input.predicted_base is not None and audit_input.predicted_intervened is not None:
            expected = audit_input.expected_affected_object_ids or []
            counterfactual_metrics, counterfactual_findings = self.contract.check_counterfactual(
                audit_input.predicted_base,
                audit_input.predicted_intervened,
                expected,
            )
            metrics.update(counterfactual_metrics)
            causal_findings = counterfactual_findings

            if "counterfactual_locality" not in counterfactual_metrics:
                metrics["counterfactual_locality"] = intervention_locality_score(
                    audit_input.predicted_base,
                    audit_input.predicted_intervened,
                    expected,
                )
                causal_findings = [
                    *causal_findings,
                    *unexpected_side_effect_findings(
                        audit_input.predicted_base,
                        audit_input.predicted_intervened,
                        expected,
                    ),
                ]

            if audit_input.intervention_description:
                for finding in causal_findings:
                    finding.details["intervention_description"] = audit_input.intervention_description
            findings.extend(causal_findings)

        has_decision_inputs = any(
            value is not None
            for value in (
                audit_input.predicted_rewards,
                audit_input.realized_rewards,
                audit_input.uncertainty,
            )
        )
        if has_decision_inputs:
            horizon = max(
                len(audit_input.predicted_rewards or []),
                len(audit_input.realized_rewards or []),
                len(audit_input.uncertainty or []),
                len(audit_input.predicted.states),
            )
            violation_by_timestep = _violation_timesteps(findings, horizon)
            metrics["decision_return_gap"] = imagined_real_return_gap(
                audit_input.predicted_rewards,
                audit_input.realized_rewards,
            )
            metrics["decision_uncertainty"] = uncertainty_on_planned_path(audit_input.uncertainty)
            metrics["decision_violation_reward_coupling"] = violation_reward_coupling(
                audit_input.predicted_rewards,
                violation_by_timestep,
            )
            if audit_input.planner_metadata is not None:
                metrics["planner_metadata"] = audit_input.planner_metadata
            decision_findings = decision_risk_findings(
                predicted_rewards=audit_input.predicted_rewards,
                realized_rewards=audit_input.realized_rewards,
                uncertainty=audit_input.uncertainty,
                violation_by_timestep=violation_by_timestep,
            )
            findings.extend(decision_findings)

        return AuditReport(
            overall_risk=_overall_risk(findings),
            findings=findings,
            metrics=metrics,
        )
