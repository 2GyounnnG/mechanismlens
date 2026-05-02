import json

from mechanismlens import AuditReport, Finding
from mechanismlens.recommendations import generate_recommendations


def make_report() -> AuditReport:
    return AuditReport(
        overall_risk="high",
        findings=[
            Finding(severity="high", category="physics", message="penetration"),
            Finding(severity="medium", category="causal", message="side effect"),
            Finding(severity="medium", category="cross_layer", message="static moved"),
        ],
        metrics={"horizon_amplification": {"h1": 0.1, "hfinal": 0.5, "hfinal_h1_ratio": 5.0}},
    )


def test_severity_counts() -> None:
    report = make_report()

    assert report.severity_counts() == {"medium": 2, "high": 1}


def test_category_counts() -> None:
    report = make_report()

    assert report.category_counts() == {"causal": 1, "physics": 1, "cross_layer": 1}


def test_markdown_contains_summary_table() -> None:
    markdown = make_report().to_markdown()

    assert "## Summary" in markdown
    assert "| Type | Low | Medium | High | Total |" in markdown
    assert "### physics" in markdown
    assert "## Recommendations" in markdown


def test_save_json_writes_valid_json(tmp_path) -> None:
    path = tmp_path / "reports" / "audit.json"

    make_report().save_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall_risk"] == "high"
    assert payload["summary"]["severity_counts"] == {"medium": 2, "high": 1}


def test_recommendations_generated_for_key_findings() -> None:
    recommendations = generate_recommendations(make_report())

    assert any("physics constraints" in recommendation for recommendation in recommendations)
    assert any("counterfactual/locality" in recommendation for recommendation in recommendations)
    assert any("semantic labels" in recommendation for recommendation in recommendations)
    assert any("short-horizon MPC" in recommendation for recommendation in recommendations)
