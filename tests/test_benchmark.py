import csv
import json

from mechanismlens.benchmark import BenchmarkRunner
from mechanismlens.examples.benchmark_cases import (
    clean_rollout,
    combined_failure,
    load_toy_benchmark_cases,
)


def test_benchmark_runner_runs_at_least_two_cases() -> None:
    results = BenchmarkRunner().run(load_toy_benchmark_cases()[:2])

    assert len(results) == 2
    assert results[0].case_name == "clean_rollout"


def test_summary_json_writes_valid_json(tmp_path) -> None:
    path = tmp_path / "summary.json"
    runner = BenchmarkRunner()
    results = runner.run(load_toy_benchmark_cases()[:2])

    runner.save_summary_json(results, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["case_name"] for item in payload] == ["clean_rollout", "penetration_failure"]


def test_summary_csv_has_expected_headers(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    runner = BenchmarkRunner()
    results = runner.run(load_toy_benchmark_cases()[:2])

    runner.save_summary_csv(results, path)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == BenchmarkRunner.CSV_HEADERS
        rows = list(reader)
    assert rows[0]["case_name"] == "clean_rollout"


def test_clean_rollout_has_lower_risk_and_fewer_findings_than_combined_failure() -> None:
    runner = BenchmarkRunner()
    clean = runner.run_case(clean_rollout())[1]
    combined = runner.run_case(combined_failure())[1]

    assert clean.overall_risk == "low"
    assert combined.overall_risk in {"medium", "high"}
    assert clean.finding_count < combined.finding_count
