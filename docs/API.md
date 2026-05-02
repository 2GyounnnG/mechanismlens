# API Overview

This page documents the small public API used by the v0.7 examples.

## ObjectState

`ObjectState` represents one object at one timestep.

```python
from mechanismlens import ObjectState

ball = ObjectState(
    object_id="ball",
    label="rigid",
    position=[0.0, 0.0],
    velocity=[1.0, 0.0],
    mass=1.0,
    radius=0.5,
)
```

## Trajectory

`Trajectory` is a list of timesteps, where each timestep is a list of `ObjectState` values.

```python
from mechanismlens import Trajectory

trajectory = Trajectory(states=[[ball]], metadata={"name": "predicted"})
```

## AuditInput

`AuditInput` packages the rollouts and optional counterfactual context.

```python
from mechanismlens import AuditInput

audit_input = AuditInput(predicted=trajectory)
```

For counterfactual audits:

```python
audit_input = AuditInput(
    predicted=intervened,
    predicted_base=base,
    predicted_intervened=intervened,
    expected_affected_object_ids=["A"],
    intervention_description="Apply an impulse to object A.",
)
```

For decision-risk audits:

```python
audit_input = AuditInput(
    predicted=trajectory,
    planned_actions=[{"action": "move_right"}],
    predicted_rewards=[1.0],
    realized_rewards=[0.2],
    uncertainty=[0.7],
    planner_metadata={"planner": "toy_greedy"},
)
```

## Finding

`Finding` is a structured diagnostic result with severity, category, message, optional time
index, and details.

```python
from mechanismlens import Finding

finding = Finding(
    severity="medium",
    category="cross_layer",
    message="Static object moved.",
    time_index=2,
)
```

## AuditReport

`AuditReport` stores findings and metrics and renders deterministic reports.

```python
report.to_markdown()
report.to_json_dict()
report.save_markdown("examples/reports/audit.md")
report.save_json("examples/reports/audit.json")
```

## AuditSuite

`AuditSuite` runs horizon metrics, domain-contract checks, counterfactual checks, and optional
decision-risk checks.

```python
from mechanismlens import AuditSuite

report = AuditSuite(bounds=[(-1, 1), (-1, 1)]).run(audit_input)
```

## DomainContract

`DomainContract` is the extension point for domain-specific audits.

```python
from typing import Any

from mechanismlens import Finding, Trajectory
from mechanismlens.contracts import DomainContract


class MyContract(DomainContract):
    name = "my_contract"
    description = "Checks for my domain."

    def check_trajectory(self, trajectory: Trajectory) -> tuple[dict[str, Any], list[Finding]]:
        return {"num_timesteps": len(trajectory.states)}, []
```

Then pass it to the suite:

```python
report = AuditSuite(contract=MyContract()).run(audit_input)
```
