import csv
import json

from mechanismlens import AuditSuite
from mechanismlens.benchmark import BenchmarkCase, BenchmarkRunner
from mechanismlens.experiments.analysis import (
    compute_detection_summary,
    severity_weighted_risk_score,
)
from mechanismlens.experiments.run_synthetic_experiment import run_experiment
from mechanismlens.experiments.synthetic_generator import (
    generate_clean_case,
    generate_combined_failure_case,
)


def test_synthetic_generator_returns_valid_case_and_labels() -> None:
    case, labels = generate_clean_case(0)

    assert isinstance(case, BenchmarkCase)
    assert labels["case_id"] == case.name
    assert labels["seed"] == 0
    assert labels["injected_failures"] == []


def test_clean_cases_have_low_or_zero_findings() -> None:
    case, _labels = generate_clean_case(1)
    report = AuditSuite().run(case.audit_input)

    assert report.overall_risk == "low"
    assert len(report.findings) == 0


def test_combined_failure_has_higher_risk_score_than_clean() -> None:
    runner = BenchmarkRunner()
    clean_case, _ = generate_clean_case(2)
    combined_case, _ = generate_combined_failure_case(2)

    clean_report, _clean_result = runner.run_case(clean_case)
    combined_report, _combined_result = runner.run_case(combined_case)

    assert severity_weighted_risk_score(combined_report) > severity_weighted_risk_score(clean_report)


def test_experiment_runner_writes_json_and_csv_for_small_n(tmp_path) -> None:
    summary = run_experiment(n_per_type=2, output_dir=tmp_path)

    records_path = tmp_path / "synthetic_audit_records.json"
    csv_path = tmp_path / "synthetic_audit_records.csv"
    summary_path = tmp_path / "synthetic_summary.json"

    assert records_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert summary["num_cases"] == 12
    assert len(json.loads(records_path.read_text(encoding="utf-8"))) == 12
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12


def test_detection_summary_contains_expected_keys() -> None:
    summary = compute_detection_summary(
        [
            {
                "injected_failures": ["physics"],
                "detected_categories": ["physics"],
            },
            {
                "injected_failures": [],
                "detected_categories": [],
            },
        ]
    )

    assert "num_cases" in summary
    assert "by_injected_failure" in summary
    assert "physics" in summary["by_injected_failure"]
    assert "clean" in summary["by_injected_failure"]
