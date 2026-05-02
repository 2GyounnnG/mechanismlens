# Audit Protocol

MechanismLens v0.1 runs a deterministic local audit:

1. Represent predicted and optional ground-truth rollouts as `Trajectory` objects.
2. Wrap them in `AuditInput`.
3. Run `AuditSuite(bounds=...).run(audit_input)`.
4. Inspect `AuditReport.metrics`, `AuditReport.findings`, or `AuditReport.to_markdown()`.

The suite currently calls:

- horizon metrics when `ground_truth` is available
- boundary, penetration, and momentum physics checks
- cross-layer semantic/physics mismatch checks

Overall risk is the maximum severity among emitted findings.

## v0.2 Counterfactual Audit

When `AuditInput.predicted_base` and `AuditInput.predicted_intervened` are both provided,
MechanismLens runs a counterfactual audit:

1. Match objects by `object_id` over shared timesteps.
2. Compute mean per-object position deviation between base and intervened rollouts.
3. Compare expected affected objects against all other objects.
4. Store locality metrics under `metrics["counterfactual_locality"]`.
5. Emit causal findings for unexpected side effects outside the expected affected set.

Severity is medium when an unexpected object changes more than the side-effect threshold and
high when its mean deviation is greater than `0.5`.
