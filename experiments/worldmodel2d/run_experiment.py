"""Run the learned 2D world-model experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

from experiments.worldmodel2d.evaluate import DEFAULT_REGIMES, evaluate_models
from experiments.worldmodel2d.train import train_models

DEFAULT_RESULTS_DIR = Path("experiments/worldmodel2d/results")
RECORDS_CSV = "worldmodel2d_records.csv"
SUMMARY_JSON = "worldmodel2d_summary.json"
SUMMARY_MD = "worldmodel2d_summary.md"


def run_experiment(
    n_train: int = 2_000,
    n_rollouts: int = 20,
    horizon: int = 16,
    seed: int = 0,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    train_torch: bool = True,
) -> dict[str, Any]:
    """Train dynamics models, audit rollouts, and write experiment outputs."""

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    models = train_models(n_train=n_train, seed=seed, train_torch=train_torch)
    records = evaluate_models(
        models=models,
        n_rollouts=n_rollouts,
        horizon=horizon,
        seed=seed + 1_000,
        regimes=DEFAULT_REGIMES,
    )
    summary = _summarize(records, models=sorted(models), regimes=list(DEFAULT_REGIMES))
    _write_records_csv(records, results_path / RECORDS_CSV)
    _write_json(summary, results_path / SUMMARY_JSON)
    _write_summary_markdown(summary, results_path / SUMMARY_MD)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Run the learned 2D world-model experiment.")
    parser.add_argument("--n-train", type=int, default=2_000)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-torch", action="store_true", help="Skip optional torch MLP training.")
    args = parser.parse_args(argv)

    summary = run_experiment(
        n_train=args.n_train,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        seed=args.seed,
        results_dir=args.results_dir,
        train_torch=not args.no_torch,
    )
    _print_summary(summary)
    return 0


def _summarize(
    records: list[dict[str, Any]],
    *,
    models: list[str],
    regimes: list[str],
) -> dict[str, Any]:
    return {
        "num_records": len(records),
        "models": models,
        "regimes": regimes,
        "mean_mse_by_model_regime": _mean_by_group(records, "mse_mean"),
        "mean_audit_risk_by_model_regime": _mean_by_group(records, "audit_risk_score"),
        "mse_audit_risk_correlation": _pearson_records(records, "mse_mean", "audit_risk_score"),
        "audit_risk_return_gap_correlation": _pearson_records(
            records, "audit_risk_score", "return_gap"
        ),
    }


def _mean_by_group(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)):
            continue
        model = str(record["model_name"])
        regime = str(record["regime"])
        totals.setdefault(model, {})
        counts.setdefault(model, {})
        totals[model][regime] = totals[model].get(regime, 0.0) + float(value)
        counts[model][regime] = counts[model].get(regime, 0) + 1
    return {
        model: {
            regime: totals[model][regime] / counts[model][regime]
            for regime in sorted(totals[model])
        }
        for model in sorted(totals)
    }


def _pearson_records(records: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    left: list[float] = []
    right: list[float] = []
    for record in records:
        left_value = record.get(left_key)
        right_value = record.get(right_key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            left.append(float(left_value))
            right.append(float(right_value))
    return _pearson(left, right)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denom = sqrt(sum(value * value for value in left_centered)) * sqrt(
        sum(value * value for value in right_centered)
    )
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denom


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Learned 2D World-Model Experiment Summary",
        "",
        f"Records: **{summary['num_records']}**",
        f"Models: **{', '.join(summary['models'])}**",
        f"Regimes: **{', '.join(summary['regimes'])}**",
        "",
        "## Mean MSE by Model/Regime",
        "",
        *_table_lines(summary["mean_mse_by_model_regime"]),
        "",
        "## Mean Audit Risk by Model/Regime",
        "",
        *_table_lines(summary["mean_audit_risk_by_model_regime"]),
        "",
        "## Correlations",
        "",
        "- `mse_audit_risk_correlation`: "
        f"{_format_optional(summary['mse_audit_risk_correlation'])}",
        "- `audit_risk_return_gap_correlation`: "
        f"{_format_optional(summary['audit_risk_return_gap_correlation'])}",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _table_lines(grouped: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| Model | Regime | Mean |", "| --- | --- | ---: |"]
    for model, regimes in grouped.items():
        for regime, value in regimes.items():
            lines.append(f"| {model} | {regime} | {value:.6f} |")
    return lines


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"models evaluated: {', '.join(summary['models'])}")
    print(f"regimes evaluated: {', '.join(summary['regimes'])}")
    print("mean MSE by model/regime:")
    _print_grouped(summary["mean_mse_by_model_regime"])
    print("mean audit risk by model/regime:")
    _print_grouped(summary["mean_audit_risk_by_model_regime"])
    print(
        "correlation MSE/audit risk: "
        f"{_format_optional(summary['mse_audit_risk_correlation'])}"
    )
    print(
        "correlation audit risk/return gap: "
        f"{_format_optional(summary['audit_risk_return_gap_correlation'])}"
    )


def _print_grouped(grouped: dict[str, dict[str, float]]) -> None:
    for model, regimes in grouped.items():
        for regime, value in regimes.items():
            print(f"  {model}/{regime}: {value:.6f}")


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
