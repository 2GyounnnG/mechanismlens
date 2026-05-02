"""Run the deterministic toy MechanismLens benchmark."""

from __future__ import annotations

from pathlib import Path

from mechanismlens.benchmark import BenchmarkRunner, BenchmarkResult
from mechanismlens.examples.benchmark_cases import load_toy_benchmark_cases

SUMMARY_JSON_PATH = Path("examples/reports/toy_benchmark_summary.json")
SUMMARY_CSV_PATH = Path("examples/reports/toy_benchmark_summary.csv")


def _print_table(results: list[BenchmarkResult]) -> None:
    print("case_name | risk | findings | high | medium | low")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for result in results:
        print(
            f"{result.case_name} | "
            f"{result.overall_risk} | "
            f"{result.finding_count} | "
            f"{result.severity_counts.get('high', 0)} | "
            f"{result.severity_counts.get('medium', 0)} | "
            f"{result.severity_counts.get('low', 0)}"
        )


def main() -> None:
    cases = load_toy_benchmark_cases()
    runner = BenchmarkRunner()
    results = runner.run(cases)
    runner.save_summary_json(results, SUMMARY_JSON_PATH)
    runner.save_summary_csv(results, SUMMARY_CSV_PATH)
    _print_table(results)


if __name__ == "__main__":
    main()
