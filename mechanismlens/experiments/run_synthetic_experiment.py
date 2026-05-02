"""Run controlled synthetic mechanism-mismatch experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mechanismlens import AuditSuite
from mechanismlens.benchmark import BenchmarkCase, BenchmarkRunner
from mechanismlens.experiments.analysis import (
    compute_category_confusion,
    compute_detection_summary,
    compute_risk_failure_correlation,
    extract_case_record,
)
from mechanismlens.experiments.synthetic_generator import (
    generate_causal_side_effect_case,
    generate_clean_case,
    generate_combined_failure_case,
    generate_cross_layer_failure_case,
    generate_decision_risk_case,
    generate_physics_failure_case,
)

Generator = Callable[[int], tuple[BenchmarkCase, dict[str, Any]]]
DEFAULT_OUTPUT_DIR = Path("experiments/results")
RECORDS_JSON = "synthetic_audit_records.json"
RECORDS_CSV = "synthetic_audit_records.csv"
SUMMARY_JSON = "synthetic_summary.json"

GENERATORS: tuple[tuple[str, Generator], ...] = (
    ("clean", generate_clean_case),
    ("physics", generate_physics_failure_case),
    ("cross_layer", generate_cross_layer_failure_case),
    ("causal", generate_causal_side_effect_case),
    ("decision", generate_decision_risk_case),
    ("combined", generate_combined_failure_case),
)


def generate_cases(n_per_type: int) -> list[tuple[BenchmarkCase, dict[str, Any]]]:
    """Generate deterministic cases for every synthetic condition."""

    cases: list[tuple[BenchmarkCase, dict[str, Any]]] = []
    for type_index, (_name, generator) in enumerate(GENERATORS):
        for case_index in range(n_per_type):
            seed = type_index * 10_000 + case_index
            cases.append(generator(seed))
    return cases


def run_experiment(
    n_per_type: int = 20,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    make_plots: bool = False,
) -> dict[str, Any]:
    """Run the synthetic experiment and write records plus summary outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(AuditSuite())
    records: list[dict[str, Any]] = []

    for case, labels in generate_cases(n_per_type):
        report, result = runner.run_case(case)
        records.append(extract_case_record(case, report, result, labels))

    summary = {
        "num_cases": len(records),
        "n_per_type": n_per_type,
        "detection_summary": compute_detection_summary(records),
        "category_confusion": compute_category_confusion(records),
        "risk_failure_correlation": compute_risk_failure_correlation(records),
        "mean_risk_score_by_injected_failure": _mean_risk_by_failure(records),
        "downstream_failure_rate_by_risk_level": _downstream_rate_by_risk(records),
    }

    _write_json(records, output_path / RECORDS_JSON)
    _write_records_csv(records, output_path / RECORDS_CSV)
    _write_json(summary, output_path / SUMMARY_JSON)

    if make_plots:
        from mechanismlens.experiments.plots import generate_plots

        generate_plots(records, output_path)

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the synthetic experiment runner."""

    parser = argparse.ArgumentParser(description="Run synthetic MechanismLens experiments.")
    parser.add_argument("--n-per-type", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots", action="store_true", help="Generate optional matplotlib plots.")
    args = parser.parse_args(argv)

    summary = run_experiment(args.n_per_type, args.output_dir, make_plots=args.plots)
    _print_summary(summary)
    return 0


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "injected_failures",
        "detected_categories",
        "severity",
        "has_downstream_failure",
        "overall_risk",
        "finding_count",
        "risk_score",
        "mean_prediction_error",
        "return_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{field: record.get(field) for field in fieldnames},
                    "injected_failures": "|".join(record.get("injected_failures", [])),
                    "detected_categories": "|".join(record.get("detected_categories", [])),
                }
            )


def _mean_risk_by_failure(records: list[dict[str, Any]]) -> dict[str, float]:
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
    print("detection counts by category:")
    by_failure = summary["detection_summary"]["by_injected_failure"]
    for category, stats in sorted(by_failure.items()):
        if stats["total"]:
            print(
                f"  {category}: detected={stats['detected']} total={stats['total']} "
                f"rate={stats['detection_rate']:.2f}"
            )
    print("mean risk score by injected failure type:")
    for label, value in summary["mean_risk_score_by_injected_failure"].items():
        print(f"  {label}: {value:.2f}")
    print("mean downstream failure rate by risk level:")
    for label, value in summary["downstream_failure_rate_by_risk_level"].items():
        print(f"  {label}: {value:.2f}")
    correlations = summary["risk_failure_correlation"]
    print(
        "correlation risk/downstream: "
        f"{_format_optional(correlations['risk_downstream_failure_correlation'])}"
    )
    print(
        "correlation risk/return_gap: "
        f"{_format_optional(correlations['risk_return_gap_correlation'])}"
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
