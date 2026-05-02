# Learned 2D World-Model Experiment

This experiment trains small supervised dynamics predictors on a deterministic 2D
environment, rolls them out, and audits the predicted trajectories with MechanismLens.

Run:

```bash
python -m experiments.worldmodel2d.run_experiment --n-train 2000 --n-rollouts 20 --horizon 16
```

The NumPy linear model always runs. If PyTorch is installed, a small MLP model is also trained;
if PyTorch is missing, the experiment continues with the linear baseline only.

Outputs are written to `experiments/worldmodel2d/results/`:

- `worldmodel2d_records.csv`
- `worldmodel2d_summary.json`
- `worldmodel2d_summary.md`
