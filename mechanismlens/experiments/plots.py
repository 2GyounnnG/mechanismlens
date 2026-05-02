"""Optional plotting helpers for synthetic experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mechanismlens.experiments.analysis import compute_category_confusion


def generate_plots(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Generate plots when matplotlib is available.

    If matplotlib is missing, this function prints a warning and returns an empty list.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib is not installed; skipping experiment plots.")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    risk_by_type = _risk_by_failure_type(records)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(risk_by_type), list(risk_by_type.values()))
    ax.set_ylabel("Mean risk score")
    ax.set_title("Risk score by injected failure type")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output_path / "risk_score_by_failure_type.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return_gap_records = [
        record for record in records if isinstance(record.get("return_gap"), (int, float))
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        [record["risk_score"] for record in return_gap_records],
        [record["return_gap"] for record in return_gap_records],
    )
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Return gap")
    ax.set_title("Risk score vs return gap")
    fig.tight_layout()
    path = output_path / "risk_vs_return_gap.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    confusion = compute_category_confusion(records)
    labels = sorted(confusion)
    matrix = [[confusion[row].get(col, 0) for col in labels] for row in labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Detected category")
    ax.set_ylabel("Injected category")
    ax.set_title("Category confusion")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = output_path / "category_confusion_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return written


def _risk_by_failure_type(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        labels = record.get("injected_failures", []) or ["clean"]
        if len(labels) > 1:
            labels = ["combined"]
        for label in labels:
            totals[label] = totals.get(label, 0.0) + float(record.get("risk_score", 0.0))
            counts[label] = counts.get(label, 0) + 1
    return {label: totals[label] / counts[label] for label in sorted(totals)}
