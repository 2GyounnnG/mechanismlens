# Synthetic Experiments

MechanismLens includes a controlled synthetic experiment suite for testing whether audit
metrics recover injected mechanism-mismatch failures and whether mechanism risk separates
low-MSE/high-risk cases from ordinary prediction error.

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
- prediction-error-only rollouts with moderate MSE but no mechanism violation
- physics failures
- cross-layer failures
- causal side effects
- decision-risk failures
- low-MSE/high-risk failures
- combined failures

Each generated case returns a `BenchmarkCase` plus ground-truth labels:

- case id and failure type
- injected failure categories
- injected severity
- expected overall risk
- downstream failure label
- seed
- whether the case is expected to have low position error

The experiment runs `AuditSuite`, extracts metrics and findings, and computes lightweight
analysis summaries without pandas or external datasets.

## Metrics

The analysis records:

- severity-weighted risk score (`low=1`, `medium=2`, `high=4`)
- finding counts by category and severity
- final and mean position error when ground truth is present
- imagined-real return gap when reward traces are present
- category recall for physics, causal, cross-layer, and decision injections
- Pearson correlations between audit risk, position error, downstream failure, and return gap
- counts of expected low-MSE cases that are still high risk

## Outputs

The default output directories are `experiments/results/` and `experiments/figures/`.

- `synthetic_audit_records.json`: full per-case records with labels, counts, risk score, position error, and return gaps.
- `synthetic_audit_records.csv`: flat per-case table for quick inspection.
- `synthetic_summary.json`: aggregate detection summaries, category recall, risk by failure type, downstream failure rates, correlations, and low-MSE/high-risk counts.
- `synthetic_summary.md`: readable Markdown summary of the same aggregate results.
- `risk_by_failure_type.png`: optional plot when matplotlib is installed.
- `risk_vs_return_gap.png`: optional plot when matplotlib is installed.
- `mse_vs_return_gap.png`: optional plot when matplotlib is installed.

If matplotlib is not installed, plotting is skipped with a warning and the experiment still
completes.

## Limitations

These cases are synthetic and intentionally small. They are not external benchmarks and do not
prove real-world safety. The purpose is to test whether MechanismLens metrics behave sensibly
when mechanism-mismatch failures are injected under controlled conditions.
