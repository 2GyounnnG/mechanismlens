import csv
import json

from mechanismlens import AuditSuite
from mechanismlens.benchmark import BenchmarkCase, BenchmarkRunner
from mechanismlens.experiments.analysis import (
    compute_detection_summary,
    compute_metric_correlations,
    severity_weighted_risk_score,
    severity_weighted_risk_score_from_counts,
)
from mechanismlens.experiments.run_synthetic_experiment import run_experiment
from mechanismlens.experiments.synthetic_generator import (
    generate_clean_case,
    generate_combined_failure_case,
    generate_low_mse_high_risk_case,
    generate_synthetic_suite,
)


def test_synthetic_generator_returns_valid_case_and_labels() -> None:
    case, labels = generate_clean_case(0)

    assert isinstance(case, BenchmarkCase)
    assert labels["case_id"] == case.name
    assert labels["failure_type"] == "clean"
    assert labels["seed"] == 0
    assert labels["injected_failures"] == []
    assert labels["expected_low_mse"] is True


def test_generate_synthetic_suite_returns_expected_number_of_cases() -> None:
    cases = generate_synthetic_suite(n_per_type=2, seed=10)

    assert len(cases) == 16
    assert len({labels["case_id"] for _case, labels in cases}) == 16


def test_clean_case_has_lower_risk_than_combined_failure() -> None:
    runner = BenchmarkRunner()
    clean_case, _ = generate_clean_case(2)
    combined_case, _ = generate_combined_failure_case(2)

    clean_report, _clean_result = runner.run_case(clean_case)
    combined_report, _combined_result = runner.run_case(combined_case)

    assert clean_report.overall_risk == "low"
    assert severity_weighted_risk_score(combined_report) > severity_weighted_risk_score(
        clean_report
    )


def test_low_mse_high_risk_case_is_labeled_and_detected() -> None:
    case, labels = generate_low_mse_high_risk_case(3)
    report = AuditSuite().run(case.audit_input)

    assert labels["expected_low_mse"] is True
    assert labels["failure_type"] == "low_mse_high_risk"
    assert report.overall_risk == "high"
    assert report.metrics["mean_position_error"] == [0.0, 0.0]
    assert severity_weighted_risk_score(report) >= 4.0


def test_analysis_risk_score_from_severity_counts() -> None:
    assert severity_weighted_risk_score_from_counts({"low": 2, "medium": 1, "high": 1}) == 8.0


def test_experiment_runner_writes_outputs_for_small_n(tmp_path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    summary = run_experiment(n_per_type=1, results_dir=results_dir, figures_dir=figures_dir)

    records_path = results_dir / "synthetic_audit_records.json"
    csv_path = results_dir / "synthetic_audit_records.csv"
    summary_path = results_dir / "synthetic_summary.json"
    summary_md_path = results_dir / "synthetic_summary.md"

    assert records_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert summary_md_path.exists()
    assert summary["num_cases"] == 8
    assert len(json.loads(records_path.read_text(encoding="utf-8"))) == 8
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert "risk_score" in rows[0]


def test_detection_summary_contains_expected_keys() -> None:
    summary = compute_detection_summary(
        [
            {
                "failure_type": "physics",
                "injected_failures": ["physics"],
                "finding_count": 1,
                "physics_count": 1,
            },
            {
                "failure_type": "clean",
                "injected_failures": [],
                "finding_count": 0,
                "physics_count": 0,
            },
        ]
    )

    assert "num_cases" in summary
    assert "by_failure_type" in summary
    assert "physics" in summary["by_failure_type"]
    assert "clean" in summary["by_failure_type"]
    assert "detected_category_counts" in summary


def test_correlation_functions_handle_missing_and_constant_values() -> None:
    correlations = compute_metric_correlations(
        [
            {"risk_score": 1.0, "has_downstream_failure": False},
            {"risk_score": 1.0, "has_downstream_failure": True},
            {"risk_score": 1.0, "return_gap": None, "mean_position_error_mean": None},
        ]
    )

    assert correlations["risk_score_vs_has_downstream_failure"] is None
    assert correlations["risk_score_vs_return_gap"] is None
