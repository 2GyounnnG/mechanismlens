"""MechanismLens audit pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from .contracts.base import DomainContract
from .contracts.toy_rigid_body import ToyRigidBodyContract
from .metrics.causal import intervention_locality_score, unexpected_side_effect_findings
from .metrics.horizon import horizon_amplification, mean_position_error
from .schema import AuditInput, AuditReport, Finding, Risk

RISK_ORDER: dict[Risk, int] = {"low": 0, "medium": 1, "high": 2}


def _overall_risk(findings: list[Finding]) -> Risk:
    if not findings:
        return "low"
    return max((finding.severity for finding in findings), key=lambda item: RISK_ORDER[item])


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

        return AuditReport(
            overall_risk=_overall_risk(findings),
            findings=findings,
            metrics=metrics,
        )
