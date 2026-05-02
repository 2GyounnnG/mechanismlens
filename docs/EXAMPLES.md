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
