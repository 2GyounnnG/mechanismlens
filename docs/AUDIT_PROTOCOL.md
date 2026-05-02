# Audit Protocol

MechanismLens runs a deterministic local audit:

1. Represent predicted and optional ground-truth rollouts as `Trajectory` objects.
2. Wrap them in `AuditInput`.
3. Choose a `DomainContract`, or use the default `ToyRigidBodyContract`.
4. Run `AuditSuite(bounds=..., contract=...).run(audit_input)`.
5. Inspect `AuditReport.metrics`, `AuditReport.findings`, or `AuditReport.to_markdown()`.

The suite currently calls:

- horizon metrics when `ground_truth` is available
- `contract.check_trajectory(predicted)` for domain-specific metrics and findings
- optional counterfactual checks when base/intervened predictions are available

Overall risk is the maximum severity among emitted findings.

## v0.3 Contract Flow

Contracts expose:

- `check_trajectory(trajectory) -> tuple[dict, list[Finding]]`
- `check_counterfactual(base, intervened, expected_affected_object_ids=None) -> tuple[dict, list[Finding]]`

The default contract is `ToyRigidBodyContract`, which wraps boundary, penetration, momentum,
and semantic/physics mismatch checks. `GenericTrajectoryContract` is safer for domains where
rigid-body assumptions are not valid.

## v0.2 Counterfactual Audit

When `AuditInput.predicted_base` and `AuditInput.predicted_intervened` are both provided,
MechanismLens runs a counterfactual audit:

1. Match objects by `object_id` over shared timesteps.
2. Compute mean per-object position deviation between base and intervened rollouts.
3. Compare expected affected objects against all other objects.
4. Store locality metrics under `metrics["counterfactual_locality"]`.
5. Emit causal findings for unexpected side effects outside the expected affected set.

Contracts may provide their own counterfactual metrics. If they do not, `AuditSuite` falls back
to MechanismLens' built-in locality score and unexpected side-effect findings.

Severity is medium when an unexpected object changes more than the side-effect threshold and
high when its mean deviation is greater than `0.5`.
