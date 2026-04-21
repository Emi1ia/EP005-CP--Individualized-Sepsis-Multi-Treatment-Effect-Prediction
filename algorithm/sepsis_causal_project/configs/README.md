# Config Layout

This folder is organized by workflow stage:

- `base/`: default and smoke configs used for standard runs.
- `tuning/`: tuning experiments and tuned snapshots.
- `optimization/`: optimize-first configs that run tune + threshold calibration + export.
- `final/`: exported train/eval-ready configs produced by optimization runs.

Common entry points:

```bash
python -m sepsis_causal.cli full-run --config configs/base/default.yaml
python -m sepsis_causal.cli tune --config configs/tuning/tune_bayes_quick.yaml
python -m sepsis_causal.cli optimize --config configs/optimization/optimize_everything.yaml
python -m sepsis_causal.cli train --config configs/final/final_optimized_for_training.yaml
```
