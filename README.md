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
report and saves `audit_report.md`.

## Counterfactual Quickstart

```bash
python -m mechanismlens.examples.toy_counterfactual_demo
```

The counterfactual demo compares a base predicted rollout against an intervened predicted
rollout. Object `A` is expected to change, while distant object `D` changes unexpectedly. The
audit reports locality metrics and causal side-effect findings, then saves
`counterfactual_audit_report.md`.

## Minimal API

```python
from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory

predicted = Trajectory(states=[
    [ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5)]
])

report = AuditSuite(bounds=[(-1.0, 1.0), (-1.0, 1.0)]).run(AuditInput(predicted=predicted))
print(report.to_markdown())
```

## Initial Failure Taxonomy

- Semantic grounding failure: labels or entities do not match the modeled world.
- Causal validity failure: interventions have implausible effects or wrong directionality.
- Physics feasibility failure: rollouts violate simple physical constraints.
- Cross-layer mechanism mismatch: semantic, causal, and physical layers contradict one another.
- Decision/planner exploitation risk: a planner chooses actions that exploit known model errors.
