# Synthetic Experiments

MechanismLens includes a controlled synthetic experiment suite for testing whether audit
metrics recover injected mechanism-mismatch failures.

Run:

```bash
python -m mechanismlens.experiments.run_synthetic_experiment --n-per-type 20
```

For a quick smoke run:

```bash
python -m mechanismlens.experiments.run_synthetic_experiment --n-per-type 5
```

## Design

The generator creates deterministic cases for:

- clean rollouts
- physics failures
- cross-layer failures
- causal side effects
- decision-risk failures
- combined failures

Each generated case returns a `BenchmarkCase` plus ground-truth labels:

- injected failure categories
- injected severity
- downstream failure label
- random seed

The experiment then runs `AuditSuite`, extracts metrics and findings, and computes lightweight
analysis summaries without pandas or external datasets.

## Outputs

The default output directory is `experiments/results/`.

- `synthetic_audit_records.json`: full per-case records with labels, metrics, findings, risk score, and return gaps.
- `synthetic_audit_records.csv`: compact per-case table for quick inspection.
- `synthetic_summary.json`: aggregate detection summaries, category confusion, mean risk by failure type, downstream failure rates, and simple correlations.

Optional plots can be requested with `--plots`. If matplotlib is not installed, plotting is
skipped with a warning.

## Limitations

These cases are synthetic and intentionally small. They are not external benchmarks and do not
prove real-world safety. The purpose is to test whether MechanismLens metrics behave sensibly
when mechanism-mismatch failures are injected under controlled conditions.
