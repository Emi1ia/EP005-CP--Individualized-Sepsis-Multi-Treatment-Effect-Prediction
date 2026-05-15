# EP005-CP--Individualized-Sepsis-Multi-Treatment-Effect-Prediction

Official implementation for individualized multi-treatment sepsis effect prediction.

This README is structured to follow the `paperswithcode/releasing-research-code` checklist:
1. dependency specification,
2. training code,
3. evaluation code,
4. pre-trained model access,
5. results table with exact commands.

## Repository Layout

- `algorithm/sepsis_causal_project/`: core pipeline code, configs, and CLI.
- `data/`: lightweight metadata/docs only (large datasets and run artifacts are local-only).
- `artifacts_release/`: compact release checkpoint bundle for fast evaluation.

## Requirements

```powershell
cd algorithm\sepsis_causal_project
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Dependencies are declared in `algorithm/sepsis_causal_project/pyproject.toml` and mirrored in
`algorithm/sepsis_causal_project/requirements.txt`.

## Dataset Setup

```powershell
cd algorithm\sepsis_causal_project
.\scripts\download_required_datasets.ps1 -DataRoot "c:\Users\emili\sepsis_project\data"
python .\scripts\check_dataset_status.py --data-root "c:\Users\emili\sepsis_project\data"
```

## Training

```powershell
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli train --config configs\final\final_patient_horizon_for_training.yaml
```

## Evaluation

```powershell
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli evaluate --config configs\final\final_patient_horizon_for_training.yaml
```

## Pre-trained Models

- Release checkpoint: `artifacts_release/final_patient_horizon/model/best_model.pt`
- Linked metrics: `artifacts_release/final_patient_horizon/eval/metrics.json`

To evaluate with this checkpoint, copy it to your configured run output model path and run `evaluate`:

```powershell
Copy-Item artifacts_release\final_patient_horizon\model\best_model.pt `
  c:\Users\emili\sepsis_project\data\sepsis_causal_artifacts\artifacts_final_patient_horizon\model\best_model.pt -Force
cd algorithm\sepsis_causal_project
python -m sepsis_causal.cli evaluate --config configs\final\final_patient_horizon_for_training.yaml
```

## Results

Metrics below were produced from each run's `eval/metrics.json`.

| Experiment | Training Command | Evaluation Command | Factual AUROC | Factual AUPRC | Factual F1 | Patient F1 | PEHE | ATE Error | Policy Regret |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final treatment policy | `python -m sepsis_causal.cli train --config configs/final/final_treatment_policy_for_training.yaml` | `python -m sepsis_causal.cli evaluate --config configs/final/final_treatment_policy_for_training.yaml` | 0.5872 | 0.2230 | 0.0243 | n/a | 0.0307 | 0.0032 | 0.0081 |
| Final goal balanced | `python -m sepsis_causal.cli train --config configs/final/final_goal_balanced_for_training.yaml` | `python -m sepsis_causal.cli evaluate --config configs/final/final_goal_balanced_for_training.yaml` | 0.5824 | 0.2183 | 0.0221 | 0.3878 | 0.0353 | 0.0098 | 0.0160 |
| Final patient horizon | `python -m sepsis_causal.cli train --config configs/final/final_patient_horizon_for_training.yaml` | `python -m sepsis_causal.cli evaluate --config configs/final/final_patient_horizon_for_training.yaml` | 0.5698 | 0.2054 | 0.0093 | 0.3837 | 0.0433 | 0.0087 | 0.0087 |

## Contributing

Open an issue or pull request with:
- the exact config path used,
- the exact command(s) run,
- and any new metrics file added under `artifacts_release/` or docs.

## License

This repository is licensed under the MIT License. See `LICENSE`.
