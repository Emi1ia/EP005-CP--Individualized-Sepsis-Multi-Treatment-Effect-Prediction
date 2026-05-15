# Metric Direction Targets (Updated)

Use these directions for training decisions and reporting:

| Metric group | Metric | Direction | Practical interpretation |
|---|---|---|---|
| Classification | AUROC | higher is better (max 1.0) | ranking quality |
| Classification | AUPRC | higher is better (max 1.0) | precision-recall quality on imbalanced data |
| Classification | F1 | higher is better (max 1.0) | thresholded precision/recall balance |
| Error | MAE | lower is better (min 0.0) | probability calibration error |
| Error | RMSE | lower is better (min 0.0) | calibration + large-error penalty |
| Recommendation error | PEHE | lower is better (min 0.0) | treatment-effect estimation error |
| Recommendation error | ATE error | lower is better (min 0.0) | average treatment-effect error |
| Recommendation error | Policy regret | lower is better (min 0.0) | policy decision loss vs oracle |

## Notes

- Error metrics should be pushed **as close to 0 as possible**.
- Classification and error metrics trade off; optimize both using weighted objective (not one metric alone).
- Always calibrate threshold after tuning for best F1 operating point.
