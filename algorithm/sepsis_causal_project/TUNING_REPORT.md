# Hyperparameter Tuning Report

## Recommendation

Use **Bayesian optimization (Optuna TPE)** for this project.

## Why

- Training each trial is expensive (sequence model on long ICU trajectories).
- Grid search is sample-inefficient in this regime.
- On the same tuning budget here, Bayesian search found a substantially better validation loss.

## Experimental comparison (quick budget)

- Bayesian study: `sepsis_hparam_tuning_quick`
  - Trials: 8
  - Best objective (`val_total`, lower is better): **2.2927**
  - Result file: `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/sepsis_hparam_tuning_quick/best_result.json`

- Grid study: `sepsis_hparam_tuning_grid_quick`
  - Trials: 9
  - Best objective (`val_total`): **2.9306**
  - Result file: `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/sepsis_hparam_tuning_grid_quick/best_result.json`

Comparison file: `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/comparison_bayes_vs_grid.json`

## Best hyperparameters from Bayesian run

- `model.hidden_size`: 96
- `model.num_heads`: 4
- `model.num_layers`: 3
- `model.ff_dim`: 192
- `model.dropout`: 0.1321
- `train.learning_rate`: 1.7087e-4
- `train.weight_decay`: 9.6398e-6
- `train.batch_size`: 64
- `train.lambda_propensity`: 0.7451
- `train.lambda_balance`: 0.2480
- `train.lambda_smooth`: 0.0745

Config ready for retraining:

- `configs/tuning/tuned_bayes_best.yaml`

## Focused Bayesian Round (latest)

- Study: `sepsis_hparam_tuning_focused_round2`
- Method: Bayesian (TPE), resumed study with `load_if_exists=True`
- Trials completed: **22**
- Best trial: **#12**
- Best objective (`val_total`, lower is better): **2.1822**
- Study artifacts:
  - `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/sepsis_hparam_tuning_focused_round2/best_result.json`
  - `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/sepsis_hparam_tuning_focused_round2/best_config_patch.json`

Best params from this round:

- `model.hidden_size`: 128
- `model.num_heads`: 8
- `model.num_layers`: 2
- `model.ff_dim`: 384
- `model.dropout`: 0.1095
- `train.learning_rate`: 1.5823e-4
- `train.weight_decay`: 1.6133e-3
- `train.batch_size`: 64
- `train.lambda_propensity`: 0.7039
- `train.lambda_balance`: 0.0729
- `train.lambda_smooth`: 0.0302

Config used for full retrain/eval:

- `configs/tuning/tuned_bayes_focused_round2_best.yaml`

## Full Evaluation Snapshot (2026-04-18)

Evaluated checkpoint:

- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts_tuned_bayes_focused_round2/model/best_model.pt`

Metrics:

- Factual: AUROC **0.5871**, AUPRC **0.2227**, F1 **0.0277**
- Treatment effect: PEHE **0.0323**, ATE error **0.0055**, Policy regret **0.0120**

Comparison summary:

- vs current baseline in `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/eval/metrics.json`:
  - Better AUROC/AUPRC/F1
  - Better PEHE and ATE error
  - Worse policy regret
- vs backup baseline in `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/model_backups/metrics_before_tuned15.json`:
  - Similar AUROC/AUPRC, better F1
  - Worse PEHE/ATE/policy regret

Comparison artifact:

- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts_tuned_bayes_focused_round2/eval/comparison_vs_baselines.json`

