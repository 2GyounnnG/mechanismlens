# Examples

MechanismLens includes two toy examples. They are deliberately small and should not be read as
claims about real-world model performance.

## Toy Rollout Audit

Run:

```bash
python -m mechanismlens.examples.toy_rollout_demo
```

The demo builds a ground-truth two-object rollout and a predicted rollout with penetration,
momentum drift, and a moving object labeled `static`.

Report snippet:

```markdown
Overall risk: **high**

| Type | Low | Medium | High | Total |
| --- | ---: | ---: | ---: | ---: |
| physics | 0 | 1 | 2 | 3 |
| cross_layer | 0 | 2 | 0 | 2 |
```

Outputs:

- `examples/reports/toy_rollout_audit.md`
- `examples/reports/toy_rollout_audit.json`

## Toy Counterfactual Audit

Run:

```bash
python -m mechanismlens.examples.toy_counterfactual_demo
```

The demo compares a base predicted rollout with an intervened predicted rollout. Object `A` is
expected to change, while object `D` changes unexpectedly.

Report snippet:

```markdown
Overall risk: **medium**

### causal

- **medium**: Unexpected side effect: object D changes by 0.350 despite not being in the expected affected set.
```

Outputs:

- `examples/reports/toy_counterfactual_audit.md`
- `examples/reports/toy_counterfactual_audit.json`

## Toy Benchmark Suite

Run:

```bash
python -m mechanismlens.examples.run_toy_benchmark
```

The benchmark suite runs multiple deterministic toy cases:

- `clean_rollout`
- `penetration_failure`
- `static_object_mismatch`
- `counterfactual_side_effect`
- `combined_failure`

It prints a compact table and writes aggregate outputs:

- `examples/reports/toy_benchmark_summary.json`
- `examples/reports/toy_benchmark_summary.csv`

The JSON file preserves metrics and count dictionaries for each case. The CSV file is intended
for quick comparison across cases and includes risk, finding count, severity counts, and category
counts.
