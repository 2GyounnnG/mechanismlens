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
- **v0.7 decision-risk audit**: imagined-real return gaps, uncertainty summaries, and
  reward/violation coupling for toy planner traces.
- **v0.8 synthetic experiment suite**: controlled mechanism-mismatch case generation,
  aggregate records, detection summaries, confusion counts, and risk/failure correlations.

## Next

- Richer contract plugins for non-rigid-body domains.
- Larger benchmark cases for planner exploitation and long-horizon rollout failures.
- Paper experiments comparing accuracy-only evaluation with synthetic cross-layer diagnostics.
- CLI entry point for batch audits and CI usage.
