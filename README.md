# MechanismLens: Cross-Layer Diagnostics for Semantic-Causal-Physical World Models

MechanismLens is a lightweight audit toolkit for inspecting predicted rollouts,
counterfactual trajectories, and planning traces from world models. It does not train models.
It helps researchers diagnose when a model's imagined future is locally plausible but wrong
because semantic grounding, causal effects, physical feasibility, or planner behavior no
longer agree.

## What Problem It Solves

World-model evaluation often emphasizes prediction error, task return, or isolated benchmark
scores. Those signals can miss cross-layer failures: an object label may be wrong even when the
motion looks plausible, an intervention may affect the wrong object, or a physically valid
rollout may be grounded on the wrong semantic entity. MechanismLens turns those assumptions
into explicit contracts and produces readable reports about where they fail.

## Core Diagnosis Categories

- **Semantic grounding**: object, entity, role, or task labels do not match the intended world.
- **Causal validity**: interventions have implausible effects, wrong locality, or wrong direction.
- **Physics feasibility**: rollouts violate boundaries, penetration constraints, momentum checks, or domain support.
- **Cross-layer mechanism mismatch**: semantic labels, causal behavior, and physical feasibility contradict each other.
- **Decision/planner risk**: planner traces may exploit model errors, uncertainty, or invalid predicted states.

## Installation

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

This project intentionally avoids deep-learning dependencies. The development extra installs
`pytest`; the core package is lightweight.

## Quickstart

Run the rollout audit demo:

```bash
python -m mechanismlens.examples.toy_rollout_demo
```

Run the counterfactual audit demo:

```bash
python -m mechanismlens.examples.toy_counterfactual_demo
```

The demos print Markdown reports and save Markdown/JSON files under `examples/reports/`.

## Benchmark Mode

Run the deterministic toy benchmark suite:

```bash
python -m mechanismlens.examples.run_toy_benchmark
```

This runs several predefined audit cases and writes aggregate outputs to:

- `examples/reports/toy_benchmark_summary.json`
- `examples/reports/toy_benchmark_summary.csv`

## Example Report Snippet

```markdown
# MechanismLens Audit Report

Overall risk: **high**

## Summary

| Type | Low | Medium | High | Total |
| --- | ---: | ---: | ---: | ---: |
| physics | 0 | 1 | 2 | 3 |
| cross_layer | 0 | 2 | 0 | 2 |
| **Total** | 0 | 3 | 2 | 5 |
```

## API Overview

`AuditInput` packages predicted, observed, ground-truth, and counterfactual trajectories for an
audit run.

`AuditSuite` orchestrates horizon metrics, domain-contract checks, counterfactual locality
metrics, and report generation.

`DomainContract` is the plugin interface for domain-specific checks. The default
`ToyRigidBodyContract` runs boundary, penetration, momentum, and semantic/physics consistency
checks.

`AuditReport` stores findings and metrics, then renders deterministic Markdown or JSON:

```python
from mechanismlens import AuditInput, AuditSuite, ObjectState, Trajectory

predicted = Trajectory(states=[
    [ObjectState("ball", label="rigid", position=[0.0, 0.0], velocity=[1.0, 0.0], mass=1.0, radius=0.5)]
])

report = AuditSuite(bounds=[(-1.0, 1.0), (-1.0, 1.0)]).run(AuditInput(predicted=predicted))
report.save_markdown("examples/reports/audit.md")
report.save_json("examples/reports/audit.json")
```

## What MechanismLens Is NOT

- It is not a world model trainer.
- It is not a universal safety proof.
- It is not a replacement for domain-specific validation or simulation review.
- It is not tied to a particular neural network framework.

## Current Limitations

- The included checks are intentionally small and rule-based.
- The rigid-body contract is a toy reference contract, not a full physics engine.
- Decision/planner risk is represented in the taxonomy but not yet deeply implemented.
- Counterfactual support currently uses object-level position deviation and locality heuristics.
- Reports explain detected contract violations; they do not establish model correctness.

## Roadmap

- v0.1: rollout, physics, horizon, and cross-layer audit basics.
- v0.2: counterfactual locality and unexpected side-effect audit.
- v0.3: `DomainContract` plugin cleanup.
- v0.4: polished Markdown/JSON reports and recommendations.
- v0.5: documentation and research framing cleanup.
- v0.6: deterministic toy benchmark suite for batch auditing.
- Next: richer contract plugins, decision-risk audit, larger benchmark cases, and paper experiments.
