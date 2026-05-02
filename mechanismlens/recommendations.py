"""Transparent rule-based recommendations for audit reports."""

from __future__ import annotations

from mechanismlens.schema import AuditReport


def generate_recommendations(report: AuditReport) -> list[str]:
    """Generate deterministic, rule-based recommendations from report findings."""

    recommendations: list[str] = []
    findings = report.findings

    if any(finding.category == "physics" and finding.severity == "high" for finding in findings):
        recommendations.append(
            "Limit rollout horizon or add physics constraints before trusting long predicted traces."
        )
    if any(finding.category == "causal" for finding in findings):
        recommendations.append(
            "Run counterfactual/locality audits with targeted intervention data around the affected objects."
        )
    if any(finding.category == "cross_layer" for finding in findings):
        recommendations.append(
            "Inspect semantic labels and check whether the active domain contract matches the modeled world."
        )
    if any(finding.category == "decision" for finding in findings):
        recommendations.append(
            "Compare imagined vs realized returns before trusting planner-selected rollouts."
        )
        recommendations.append(
            "Add an uncertainty penalty or trust-region planning constraint for risky imagined paths."
        )
        recommendations.append(
            "Avoid using long rollout reward as the sole planner objective."
        )

    amplification = report.metrics.get("horizon_amplification")
    if isinstance(amplification, dict):
        ratio = amplification.get("hfinal_h1_ratio")
        hfinal = amplification.get("hfinal")
        h1 = amplification.get("h1")
        high_amplification = isinstance(ratio, (int, float)) and ratio > 2.0
        zero_to_nonzero = h1 == 0 and isinstance(hfinal, (int, float)) and hfinal > 0
        if high_amplification or zero_to_nonzero:
            recommendations.append(
                "Prefer short-horizon MPC or uncertainty-aware rollout when horizon error amplifies."
            )

    return recommendations
