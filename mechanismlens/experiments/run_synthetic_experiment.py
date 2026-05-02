"""Run controlled synthetic mechanism-mismatch experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mechanismlens import AuditSuite
from mechanismlens.benchmark import BenchmarkRunner
from mechanismlens.experiments.analysis import (
    compute_category_recall,
    compute_detection_summary,
    compute_metric_correlations,
    compute_risk_by_failure_type,
    extract_case_record,
    summarize_low_mse_high_risk,
)
from mechanismlens.experiments.synthetic_generator import generate_synthetic_suite

DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_FIGURES_DIR = Path("experiments/figures")
RECORDS_JSON = "synthetic_audit_records.json"
RECORDS_CSV = "synthetic_audit_records.csv"
SUMMARY_JSON = "synthetic_summary.json"
SUMMARY_MD = "synthetic_summary.md"


def run_experiment(
    n_per_type: int = 20,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the synthetic suite and write records, summary, and optional figures."""

    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(AuditSuite())
    records: list[dict[str, Any]] = []
    cases = generate_synthetic_suite(n_per_type=n_per_type, seed=seed)

    for case, labels in cases:
        report, result = runner.run_case(case)
        records.append(extract_case_record(case, report, result, labels))

    summary = {
        "num_cases": len(records),
        "n_per_type": n_per_type,
        "seed": seed,
        "detection_summary": compute_detection_summary(records),
        "category_recall": compute_category_recall(records),
        "risk_by_failure_type": compute_risk_by_failure_type(records),
        "downstream_failure_rate_by_risk_level": _downstream_rate_by_risk(records),
        "correlations": compute_metric_correlations(records),
        "low_mse_high_risk": summarize_low_mse_high_risk(records),
    }

    _write_json(records, results_path / RECORDS_JSON)
    _write_records_csv(records, results_path / RECORDS_CSV)
    _write_json(summary, results_path / SUMMARY_JSON)
    _write_summary_markdown(summary, results_path / SUMMARY_MD)

    from mechanismlens.experiments.plots import (
        plot_mse_vs_return_gap,
        plot_risk_by_failure_type,
        plot_risk_vs_return_gap,
    )

    plot_risk_by_failure_type(records, figures_path / "risk_by_failure_type.png")
    plot_risk_vs_return_gap(records, figures_path / "risk_vs_return_gap.png")
    plot_mse_vs_return_gap(records, figures_path / "mse_vs_return_gap.png")

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the synthetic experiment runner."""

    parser = argparse.ArgumentParser(description="Run synthetic MechanismLens experiments.")
    parser.add_argument("--n-per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args(argv)

    summary = run_experiment(
        n_per_type=args.n_per_type,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
    )
    _print_summary(summary)
    return 0


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic Experiment Summary",
        "",
        f"Cases: **{summary['num_cases']}**",
        f"Cases per type: **{summary['n_per_type']}**",
        "",
        "## Mean Risk by Failure Type",
        "",
        "| Failure type | Mean risk | Median risk | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for failure_type, stats in summary["risk_by_failure_type"].items():
        lines.append(
            f"| {failure_type} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['count']} |"
        )
    lines.extend(["", "## Category Recall", ""])
    lines.extend(["| Category | Detected | Total | Recall |", "| --- | ---: | ---: | ---: |"])
    for category, stats in summary["category_recall"].items():
        lines.append(
            f"| {category} | {stats['detected']} | {stats['total']} | {stats['recall']:.2f} |"
        )
    lines.extend(["", "## Correlations", ""])
    for name, value in summary["correlations"].items():
        lines.append(f"- `{name}`: {_format_optional(value)}")
    low_mse = summary["low_mse_high_risk"]
    lines.extend(
        [
            "",
            "## Low-MSE / High-Risk Cases",
            "",
            f"{low_mse['low_mse_high_risk_count']} of "
            f"{low_mse['expected_low_mse_count']} expected low-MSE cases were high risk.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _downstream_rate_by_risk(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    failures: dict[str, int] = {}
    for record in records:
        risk = str(record.get("overall_risk", "unknown"))
        totals[risk] = totals.get(risk, 0) + 1
        failures[risk] = failures.get(risk, 0) + int(bool(record.get("has_downstream_failure")))
    return {risk: failures[risk] / totals[risk] for risk in sorted(totals)}


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"cases: {summary['num_cases']}")
    print("mean risk by failure type:")
    for failure_type, stats in summary["risk_by_failure_type"].items():
        print(f"  {failure_type}: mean={stats['mean']:.2f} median={stats['median']:.2f}")
    print("category recall:")
    for category, stats in summary["category_recall"].items():
        print(f"  {category}: {stats['detected']}/{stats['total']} ({stats['recall']:.2f})")
    print("correlations:")
    for name, value in summary["correlations"].items():
        print(f"  {name}: {_format_optional(value)}")
    low_mse = summary["low_mse_high_risk"]
    print(
        "low-MSE/high-risk cases: "
        f"{low_mse['low_mse_high_risk_count']}/{low_mse['expected_low_mse_count']}"
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
