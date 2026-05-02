# Roadmap

## v0.1

- Typed object and trajectory schema.
- Horizon position-error metrics.
- Toy rigid-body physics checks.
- Cross-layer semantic/physics mismatch checks.
- Markdown and JSON-style reports.
- Runnable toy demo and pytest coverage.

## v0.2

- Implemented counterfactual audit support for base/intervened predicted rollouts.
- Implemented per-object deviation, intervention locality metrics, and unexpected side-effect findings.
- In progress: richer causal graph auditing.

## v0.3

- Implemented contract plugin cleanup through the `DomainContract` interface.
- Implemented `ToyRigidBodyContract` and `GenericTrajectoryContract`.
- In progress: richer domain-contract registry.
- Decision-risk metrics for planning traces.

## v0.4

- Implemented polished Markdown/JSON reports with category and severity summaries.
- Implemented rule-based recommendations for physics, causal, cross-layer, and horizon risks.
- Standardized demo report outputs under `examples/reports/`.

## Later

- Optional visualization utilities.
- CLI entry point for CI and batch audits.
- Expanded benchmarks for synthetic rollout failures.
