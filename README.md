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

For CLI-only use, an editable install without development extras is enough:

```bash
python -m pip install -e .
```

## Quickstart

Run the rollout audit demo:

```bash
python -m mechanismlens.examples.toy_rollout_demo
```

Run the counterfactual audit demo:

```bash
python -m mechanismlens.examples.toy_counterfactual_demo
```

Run the decision-risk audit demo:

```bash
python -m mechanismlens.examples.toy_decision_risk_demo
```

The demos print Markdown reports and save Markdown/JSON files under `examples/reports/`.

## CLI Usage

After installation, the `mechanismlens` command exposes the same demos and benchmark:

```bash
mechanismlens demo rollout
mechanismlens demo counterfactual
mechanismlens demo decision
mechanismlens benchmark toy
mechanismlens version
```

## Benchmark Mode

Run the deterministic toy benchmark suite:

```bash
python -m mechanismlens.examples.run_toy_benchmark
```

This runs several predefined audit cases and writes aggregate outputs to:

- `examples/reports/toy_benchmark_summary.json`
- `examples/reports/toy_benchmark_summary.csv`

## Experimental Analysis

Run controlled synthetic mechanism-mismatch experiments:

```bash
python -m mechanismlens.experiments.run_synthetic_experiment --n-per-type 20
```

The experiment injects clean, prediction-error-only, physics, cross-layer, causal, decision,
low-MSE/high-risk, and combined cases, then writes:

- `experiments/results/synthetic_audit_records.json`
- `experiments/results/synthetic_audit_records.csv`
- `experiments/results/synthetic_summary.json`
- `experiments/results/synthetic_summary.md`

If matplotlib is available, it also saves optional figures under `experiments/figures/`.

These outputs are intended for research analysis: whether audit scores recover injected
failure categories, whether low prediction error can still hide mechanism risk, and whether
mechanism risk predicts downstream failure better than simple prediction error in the synthetic
setting.

Run the learned 2D world-model experiment:

```bash
python -m experiments.worldmodel2d.run_experiment --n-train 2000 --n-rollouts 20 --horizon 16
```

This trains a NumPy linear dynamics predictor, optionally trains a small PyTorch MLP when torch
is installed, rolls each model out in ID, boundary, collision, and planner regimes, then audits
the learned predicted trajectories. Outputs are written under `experiments/worldmodel2d/results/`.

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
- Decision/planner risk checks are toy-level heuristics, not a full planning safety analysis.
- Counterfactual support currently uses object-level position deviation and locality heuristics.
- Reports explain detected contract violations; they do not establish model correctness.

## Roadmap

- v0.1: rollout, physics, horizon, and cross-layer audit basics.
- v0.2: counterfactual locality and unexpected side-effect audit.
- v0.3: `DomainContract` plugin cleanup.
- v0.4: polished Markdown/JSON reports and recommendations.
- v0.5: documentation and research framing cleanup.
- v0.6: deterministic toy benchmark suite for batch auditing.
- v0.7: decision-risk audit for planner traces.
- v0.8: CLI interface for demos, benchmarks, and version checks.
- v0.9: synthetic experiment analysis for injected failure recovery and risk/error comparison.
- Next: richer contract plugins, larger benchmark cases, and paper experiments.
