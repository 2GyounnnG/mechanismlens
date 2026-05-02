"""Optional plotting helpers for synthetic experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mechanismlens.experiments.analysis import compute_risk_by_failure_type


def plot_risk_by_failure_type(records: list[dict[str, Any]], path: str | Path) -> Path | None:
    """Save a bar chart of mean risk score by failure type if matplotlib is installed."""

    plt = _matplotlib()
    if plt is None:
        return None
    output_path = _prepare_path(path)
    risk_by_type = compute_risk_by_failure_type(records)
    labels = list(risk_by_type)
    values = [risk_by_type[label]["mean"] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean risk score")
    ax.set_title("Risk score by synthetic failure type")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_risk_vs_return_gap(records: list[dict[str, Any]], path: str | Path) -> Path | None:
    """Save a risk-vs-return-gap scatter plot if matplotlib is installed."""

    plt = _matplotlib()
    if plt is None:
        return None
    output_path = _prepare_path(path)
    points = [
        (float(record["risk_score"]), float(record["return_gap"]))
        for record in records
        if isinstance(record.get("risk_score"), (int, float))
        and isinstance(record.get("return_gap"), (int, float))
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([left for left, _right in points], [right for _left, right in points])
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Return gap")
    ax.set_title("Risk vs return gap")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_mse_vs_return_gap(records: list[dict[str, Any]], path: str | Path) -> Path | None:
    """Save an MSE-vs-return-gap scatter plot if matplotlib is installed."""

    plt = _matplotlib()
    if plt is None:
        return None
    output_path = _prepare_path(path)
    points = [
        (float(record["mean_position_error_mean"]), float(record["return_gap"]))
        for record in records
        if isinstance(record.get("mean_position_error_mean"), (int, float))
        and isinstance(record.get("return_gap"), (int, float))
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([left for left, _right in points], [right for _left, right in points])
    ax.set_xlabel("Mean position error")
    ax.set_ylabel("Return gap")
    ax.set_title("Prediction error vs return gap")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _matplotlib() -> Any | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib is not installed; skipping experiment plot.")
        return None
    return plt


def _prepare_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
