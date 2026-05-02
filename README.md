# MechanismLens

MechanismLens is a modular audit toolkit for diagnosing mechanism-alignment failures in
world-model rollouts. It does **not** train world models. It audits predicted trajectories,
counterfactual trajectories, and planning traces for simple signs of mechanism mismatch.

## What It Audits

- Semantic grounding: wrong object, entity, or task labels.
- Causal validity: intervention effects appearing on the wrong objects.
- Physics feasibility: impossible dynamics, penetration, boundary violations, or drift.
- Cross-layer consistency: semantic labels contradict physical behavior.
- Decision risk: planner traces that may exploit model errors or uncertainty.

## What It Does Not Do

- It does not train neural networks.
- It does not require external datasets.
- It does not depend on PyTorch.
- It does not claim to prove full model correctness.

## Quickstart

```bash
python -m mechanismlens.examples.toy_rollout_demo
```

The demo constructs a ground-truth two-object rollout and a predicted rollout with object
penetration, a moving object labeled `static`, and momentum drift. It prints a Markdown audit
report and saves `examples/reports/toy_rollout_audit.md` and
`examples/reports/toy_rollout_audit.json`.

## Counterfactual Quickstart

```bash
python -m mechanismlens.examples.toy_counterfactual_demo
```

The counterfactual demo compares a base predicted rollout against an intervened predicted
rollout. Object `A` is expected to change, while distant object `D` changes unexpectedly. The
audit reports locality metrics and causal side-effect findings, then saves
`examples/reports/toy_counterfactual_audit.md` and
`examples/reports/toy_counterfactual_audit.json`.

## Minimal API

```python
from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory

predicted = Trajectory(states=[
    [ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5)]
])

report = AuditSuite(bounds=[(-1.0, 1.0), (-1.0, 1.0)]).run(AuditInput(predicted=predicted))
print(report.to_markdown())
```

## DomainContract Plugins

`AuditSuite` delegates domain-specific checks to a `DomainContract`. By default it uses
`ToyRigidBodyContract(bounds=...)`, which runs boundary, penetration, momentum, and
semantic/physics consistency checks. You can pass a different contract for another domain:

```python
from mechanismlens import AuditSuite
from mechanismlens.contracts import GenericTrajectoryContract

suite = AuditSuite(contract=GenericTrajectoryContract())
```

A contract implements `check_trajectory(...)` and can optionally implement
`check_counterfactual(...)`. This keeps MechanismLens framework-general while letting each
domain define the checks that actually make sense.

## Reports

Markdown reports include the overall risk, a summary table, metrics, findings grouped by
category, and rule-based recommendations when findings exist. The summary table counts findings
by category and severity (`low`, `medium`, `high`), with the total row showing the whole report.

Current finding categories are:

- `semantic`
- `causal`
- `physics`
- `cross_layer`
- `decision`
- `horizon`

Reports can be saved as Markdown and JSON:

```python
report.save_markdown("examples/reports/audit.md")
report.save_json("examples/reports/audit.json")
```

## Initial Failure Taxonomy

- Semantic grounding failure: labels or entities do not match the modeled world.
- Causal validity failure: interventions have implausible effects or wrong directionality.
- Physics feasibility failure: rollouts violate simple physical constraints.
- Cross-layer mechanism mismatch: semantic, causal, and physical layers contradict one another.
- Decision/planner exploitation risk: a planner chooses actions that exploit known model errors.
