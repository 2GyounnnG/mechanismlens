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
