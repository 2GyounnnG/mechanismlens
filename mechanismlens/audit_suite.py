"""MechanismLens audit pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from .metrics.causal import intervention_locality_score, unexpected_side_effect_findings
from .metrics.cross_layer import semantic_physics_mismatch
from .metrics.horizon import horizon_amplification, mean_position_error
from .metrics.physics import boundary_violation, momentum_drift, penetration_violation
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
    ) -> None:
        self.bounds = bounds

    def run(self, audit_input: AuditInput) -> AuditReport:
        """Run horizon, physics, cross-layer, and optional counterfactual checks."""

        findings: list[Finding] = []
        metrics: dict[str, Any] = {}

        if audit_input.ground_truth is not None:
            errors = mean_position_error(audit_input.predicted, audit_input.ground_truth)
            metrics["mean_position_error"] = errors
            metrics["horizon_amplification"] = horizon_amplification(errors)

        boundary_metric, boundary_findings = boundary_violation(audit_input.predicted, self.bounds)
        penetration_metric, penetration_findings = penetration_violation(audit_input.predicted)
        momentum_metric, momentum_findings = momentum_drift(audit_input.predicted)
        cross_layer_findings = semantic_physics_mismatch(audit_input.predicted)

        metrics["boundary_violation"] = boundary_metric
        metrics["penetration_violation"] = penetration_metric
        metrics["momentum_drift"] = momentum_metric

        findings.extend(boundary_findings)
        findings.extend(penetration_findings)
        findings.extend(momentum_findings)
        findings.extend(cross_layer_findings)

        if audit_input.predicted_base is not None and audit_input.predicted_intervened is not None:
            expected = audit_input.expected_affected_object_ids or []
            metrics["counterfactual_locality"] = intervention_locality_score(
                audit_input.predicted_base,
                audit_input.predicted_intervened,
                expected,
            )
            causal_findings = unexpected_side_effect_findings(
                audit_input.predicted_base,
                audit_input.predicted_intervened,
                expected,
            )
            if audit_input.intervention_description:
                for finding in causal_findings:
                    finding.details["intervention_description"] = audit_input.intervention_description
            findings.extend(causal_findings)

        return AuditReport(
            overall_risk=_overall_risk(findings),
            findings=findings,
            metrics=metrics,
        )
