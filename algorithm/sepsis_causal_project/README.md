# Sepsis Causal Transformer Project

This project implements an end-to-end pipeline for the proposal:

- ingest PhysioNet 2019 Sepsis Challenge patient time-series,
- run dataset quality control,
- build hourly sequence features with missingness indicators and time-since-last-observation channels,
- generate multi-treatment trajectories (antibiotics, fluids, vasopressors) using a semi-synthetic simulator,
- train a causal-transformer style model with treatment propensity, representation balancing, and outcome heads,
- evaluate factual prediction and treatment-effect quality (PEHE, ATE error, policy regret).

## Important data constraint

The PhysioNet 2019 challenge files do **not** include explicit treatment administration variables for antibiotics, fluids, or vasopressors.  
To satisfy the proposal's multi-treatment objective end-to-end in this environment, this implementation uses a semi-synthetic treatment/outcome generator conditioned on patient state.

Use real treatment-effect claims only after replacing the simulator with real treatment tables (for example from MIMIC-IV medication/inputevent pipelines).

## Layout

- `configs/base/default.yaml`: default hyperparameters and paths
- `configs/README.md`: config family index (`base`, `tuning`, `optimization`, `final`)
- `src/sepsis_causal/cli.py`: command-line entry point
- `src/sepsis_causal/qc.py`: dataset quality report
- `src/sepsis_causal/prepare.py`: preprocessing + split + semi-synthetic data build
- `src/sepsis_causal/model.py`: causal transformer model
- `src/sepsis_causal/train.py`: model training
- `src/sepsis_causal/evaluate.py`: evaluation metrics

## Setup

```bash
cd c:/Users/emili/sepsis_project/algorithm/sepsis_causal_project
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

## Download Required Datasets

From PowerShell:

```powershell
cd c:\Users\emili\sepsis_project\algorithm\sepsis_causal_project

# Set PhysioNet credentials for restricted datasets (MIMIC-IV)
$env:PHYSIONET_USERNAME = "your_username"
$env:PHYSIONET_PASSWORD = "your_password"

.\scripts\download_required_datasets.ps1 -DataRoot "c:\Users\emili\sepsis_project\data"
```

Optional ED module:

```powershell
.\scripts\download_required_datasets.ps1 -DataRoot "c:\Users\emili\sepsis_project\data" -IncludeMimicEd
```

Open-only sync (skip restricted MIMIC):

```powershell
.\scripts\download_required_datasets.ps1 -DataRoot "c:\Users\emili\sepsis_project\data" -SkipRestricted
```

Check status:

```powershell
python .\scripts\check_dataset_status.py --data-root "c:\Users\emili\sepsis_project\data"
```

## Run end to end

```bash
python -m sepsis_causal.cli full-run ^
  --config configs/base/default.yaml ^
  --data-root c:/Users/emili/sepsis_project/data/challenge-2019/training ^
  --out-dir c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts ^
  --max-patients 4000
```

Remove `--max-patients` to process all patients (long runtime).

## Individual steps

```bash
python -m sepsis_causal.cli qc --config configs/base/default.yaml
python -m sepsis_causal.cli prepare --config configs/base/default.yaml
python -m sepsis_causal.cli train --config configs/base/default.yaml
python -m sepsis_causal.cli evaluate --config configs/base/default.yaml
python -m sepsis_causal.cli tune --config configs/base/default.yaml
python -m sepsis_causal.cli optimize --config configs/optimization/optimize_everything.yaml
```

## Hyperparameter Tuning

Recommended for this project: **Bayesian optimization (Optuna TPE)**.

Reason: model training is expensive (sequence transformer + large cohort), and Bayesian search finds good regions with far fewer trials than grid search.

Quick Bayesian run:

```bash
python -m sepsis_causal.cli tune --config configs/tuning/tune_bayes_quick.yaml
```

Quick grid baseline:

```bash
python -m sepsis_causal.cli tune --config configs/tuning/tune_grid_quick.yaml
```

See `TUNING_REPORT.md` for a concrete Bayesian-vs-grid comparison and recommended tuned config.

Tuning outputs:

- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/<study_name>/best_result.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/<study_name>/best_config_patch.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/<study_name>/trials.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/tuning/<study_name>/optuna_study.db`

## Optimize First, Train Later

If you want to finish optimization first and only train afterward, use:

```bash
python -m sepsis_causal.cli optimize --config configs/optimization/optimize_everything.yaml
```

This runs tuning, calibrates a validation threshold for F1, and writes a final trainable config:

- `configs/final/final_optimized_for_training.yaml`

Then run full training/evaluation with that exported config:

```bash
python -m sepsis_causal.cli train --config configs/final/final_optimized_for_training.yaml
python -m sepsis_causal.cli evaluate --config configs/final/final_optimized_for_training.yaml
```

## Patient-Level / Horizon Sepsis Mode

For stronger classification signal, use horizon labels and patient-level scoring:

```bash
python -m sepsis_causal.cli optimize --config configs/optimization/optimize_patient_horizon.yaml
python -m sepsis_causal.cli train --config configs/final/final_patient_horizon_for_training.yaml
python -m sepsis_causal.cli evaluate --config configs/final/final_patient_horizon_for_training.yaml
```

This mode adds:

- `train.sepsis_target_mode` (`current` or `future_horizon`)
- `train.sepsis_horizon_hours`
- `eval.sepsis_target_mode`
- `eval.sepsis_horizon_hours`
- `eval.sepsis_patient_aggregation` (`max`, `mean`, `last`)
- `eval.sepsis_patient_threshold`

### Optional Training Balance Sampler

To upsample septic patients in training without changing raw files:

- `train.sampler_mode: "balanced_patient"`
- `train.sampler_positive_fraction` (for example `0.5`)
- `train.sampler_num_samples` (`null` uses one epoch worth of samples)
- `train.sampler_replacement` (`true` recommended)

## Outputs

- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/qc/quality_report.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/prepared/{train,val,test}.csv`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/prepared/patients/*.npz`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/prepared/normalization_stats.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/model/best_model.pt`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/model/train_history.json`
- `c:/Users/emili/sepsis_project/data/sepsis_causal_artifacts/artifacts/eval/metrics.json`

