import subprocess
import sys

import pytest

from mechanismlens import cli
from mechanismlens.examples import (
    run_toy_benchmark,
    toy_counterfactual_demo,
    toy_decision_risk_demo,
    toy_rollout_demo,
)


def test_python_module_cli_version_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mechanismlens.cli", "version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip()


def test_cli_demo_subcommands_call_expected_mains(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(toy_rollout_demo, "main", lambda: called.append("rollout"))
    monkeypatch.setattr(toy_counterfactual_demo, "main", lambda: called.append("counterfactual"))
    monkeypatch.setattr(toy_decision_risk_demo, "main", lambda: called.append("decision"))

    assert cli.main(["demo", "rollout"]) == 0
    assert cli.main(["demo", "counterfactual"]) == 0
    assert cli.main(["demo", "decision"]) == 0
    assert called == ["rollout", "counterfactual", "decision"]


def test_cli_benchmark_subcommand_calls_expected_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(run_toy_benchmark, "main", lambda: called.append("toy"))

    assert cli.main(["benchmark", "toy"]) == 0
    assert called == ["toy"]


def test_cli_invalid_command_raises_system_exit() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["demo", "missing"])

    assert exc_info.value.code != 0
