# Roadmap

## Implemented

- **v0.1 rollout audit**: typed trajectory schema, horizon position error, toy physics checks,
  cross-layer mismatch checks, reports, demos, and tests.
- **v0.2 counterfactual audit**: base/intervened rollout comparison, per-object deviation,
  locality metrics, and unexpected side-effect findings.
- **v0.3 contract plugins**: `DomainContract` interface plus toy rigid-body and generic
  trajectory contracts.
- **v0.4 report polish**: deterministic Markdown/JSON reports, summary tables, grouped
  findings, recommendations, and standardized example report paths.
- **v0.5 docs/research framing**: polished README, paper-style framing, API docs, examples,
  development guide, and clearer project roadmap.
- **v0.6 benchmark suite**: deterministic toy benchmark cases plus aggregate JSON/CSV outputs
  for batch auditing.

## Next

- Richer contract plugins for non-rigid-body domains.
- Decision-risk audit for planner traces.
- Paper experiments that compare accuracy-only evaluation with cross-layer diagnostics.
- CLI entry point for batch audits and CI usage.
