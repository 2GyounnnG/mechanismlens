# Development

## Setup

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

If your shell does not expose `python`, use the interpreter from your virtual environment.

## Running Tests

```bash
pytest
```

## Running Demos

```bash
python -m mechanismlens.examples.toy_rollout_demo
python -m mechanismlens.examples.toy_counterfactual_demo
```

Reports are written under `examples/reports/`.

## Branch Workflow

Use feature branches for versioned changes, for example:

```bash
git checkout -b v0.8-my-feature
```

Keep changes scoped, run tests before review, and avoid committing generated report outputs
unless a release process explicitly asks for them.

## Adding a DomainContract

Create a class that extends `DomainContract`:

```python
from typing import Any

from mechanismlens import Finding, Trajectory
from mechanismlens.contracts import DomainContract


class GridWorldContract(DomainContract):
    name = "grid_world"
    description = "Grid-world movement and obstacle checks."

    def check_trajectory(self, trajectory: Trajectory) -> tuple[dict[str, Any], list[Finding]]:
        metrics: dict[str, Any] = {"num_timesteps": len(trajectory.states)}
        findings: list[Finding] = []
        return metrics, findings
```

Pass the contract to `AuditSuite(contract=GridWorldContract())`.

## Design Principles

- Transparent rule-based checks are preferred over opaque scoring.
- No hidden network calls.
- Domain contracts are specialized and explicit.
- The toolkit is framework-general but measurement-specific.
- MechanismLens audits existing outputs; it does not train world models.
