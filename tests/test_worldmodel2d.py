import csv
import json

import numpy as np

from experiments.worldmodel2d.dataset import generate_transition_dataset
from experiments.worldmodel2d.env import WorldModel2DEnv
from experiments.worldmodel2d.evaluate import evaluate_models
from experiments.worldmodel2d.models import LinearDynamicsModel
from experiments.worldmodel2d.run_experiment import run_experiment


def test_env_step_keeps_ground_truth_inside_bounds() -> None:
    env = WorldModel2DEnv()
    env.reset(seed=0, regime="boundary")

    for _ in range(20):
        state = env.step([1.0, 0.4])
        assert -1.0 <= state[0] <= 1.0
        assert -1.0 <= state[1] <= 1.0


def test_dataset_generation_shapes_are_correct() -> None:
    x, y = generate_transition_dataset(n=25, seed=1)

    assert x.shape == (25, 6)
    assert y.shape == (25, 4)


def test_linear_dynamics_model_fit_predict_works() -> None:
    x, y = generate_transition_dataset(n=100, seed=2)
    model = LinearDynamicsModel()
    model.fit(x, y)

    prediction = model.predict_next(x[0, :4], x[0, 4:])

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))


def test_evaluation_produces_records_with_mse_and_audit_risk() -> None:
    x, y = generate_transition_dataset(n=120, seed=3)
    model = LinearDynamicsModel()
    model.fit(x, y)

    records = evaluate_models({"linear": model}, n_rollouts=1, horizon=4, regimes=("id",))

    assert len(records) == 1
    assert "mse_mean" in records[0]
    assert "audit_risk_score" in records[0]
    assert records[0]["model_name"] == "linear"


def test_worldmodel2d_run_experiment_small_settings(tmp_path) -> None:
    summary = run_experiment(
        n_train=100,
        n_rollouts=2,
        horizon=4,
        results_dir=tmp_path,
        train_torch=False,
    )

    records_path = tmp_path / "worldmodel2d_records.csv"
    summary_json_path = tmp_path / "worldmodel2d_summary.json"
    summary_md_path = tmp_path / "worldmodel2d_summary.md"

    assert records_path.exists()
    assert summary_json_path.exists()
    assert summary_md_path.exists()
    assert summary["num_records"] == 8
    with records_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert "mse_mean" in rows[0]
    assert "audit_risk_score" in rows[0]
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert "mse_audit_risk_correlation" in payload
