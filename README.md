# EP005-CP--Individualized-Sepsis-Multi-Treatment-Effect-Prediction

This project predicts sepsis risk and treatment-effect outcomes using a Causal Transformer model.

## Project layout

- `algorithm/sepsis_causal_project/`: training/evaluation code
- `artifacts_release/`: lightweight release artifacts and result snapshots
- `data/`: local data and local run outputs (not synced to GitHub)

## Specification of dependencies

Use Python 3.10+.

```powershell
cd algorithm\sepsis_causal_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You can also use:

```powershell
pip install -e .
```

## Training code

```powershell
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli train --config configs\final\final_patient_horizon_for_training.yaml
```

## Evaluation code

```powershell
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli evaluate --config configs\final\final_patient_horizon_for_training.yaml
```

## Pre-trained models

- Checkpoint: `artifacts_release/final_patient_horizon/model/best_model.pt`
- Metrics file: `artifacts_release/final_patient_horizon/eval/metrics.json`

For training, please run:

```powershell
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli train --config configs\final\final_patient_horizon_for_training.yaml
python -m sepsis_causal.cli evaluate --config configs\final\final_patient_horizon_for_training.yaml
```

Export result numbers from the produced `eval/metrics.json`:

```powershell
python algorithm\sepsis_causal_project\scripts\print_table2_results.py `
  --metrics data\sepsis_causal_artifacts\artifacts_final_patient_horizon\eval\metrics.json `
  --percent
```

### Example output table format

| Model Type | AUROC (S) | AUROC (P) | AUPRC | F1 | MAE | RMSE | PEHE | ATE Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Causal Transformer | from run | from run | from run | from run | from run | from run | from run | from run |