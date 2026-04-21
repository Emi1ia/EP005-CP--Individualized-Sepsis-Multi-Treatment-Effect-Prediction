from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatientSequenceDataset, collate_patient_batch
from .metrics import probability_error_metrics, safe_classification_metrics, treatment_effect_metrics
from .model import CausalTransformer
from .targets import build_temporal_target_torch
from .utils import save_json


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def run_evaluate(config: dict[str, Any], out_dir: Path) -> Path:
    prepared_dir = out_dir / "prepared"
    model_path = out_dir / "model" / "best_model.pt"
    test_csv = prepared_dir / "test.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test split: {test_csv}")

    ckpt = torch.load(model_path, map_location="cpu")
    mcfg = ckpt["model_config"]
    model = CausalTransformer(
        input_dim=int(mcfg["input_dim"]),
        hidden_size=int(mcfg["hidden_size"]),
        num_layers=int(mcfg["num_layers"]),
        num_heads=int(mcfg["num_heads"]),
        ff_dim=int(mcfg["ff_dim"]),
        dropout=float(mcfg["dropout"]),
    )
    state_dict = ckpt["model_state_dict"]
    has_sepsis_head = ("sepsis_head.weight" in state_dict and "sepsis_head.bias" in state_dict)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds_test = PatientSequenceDataset(test_csv)
    loader = DataLoader(
        ds_test,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate_patient_batch,
    )

    y_true = []
    y_prob = []
    y_sepsis_true = []
    y_sepsis_prob = []
    patient_sepsis_true = []
    patient_sepsis_prob = []
    pred_all = []
    true_all = []

    eval_cfg = config.get("eval", {})
    train_cfg = config.get("train", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    sepsis_threshold = float(eval_cfg.get("sepsis_threshold", threshold))
    sepsis_patient_threshold = float(eval_cfg.get("sepsis_patient_threshold", sepsis_threshold))
    sepsis_target_mode = str(eval_cfg.get("sepsis_target_mode", train_cfg.get("sepsis_target_mode", "current"))).lower()
    sepsis_horizon_hours = int(eval_cfg.get("sepsis_horizon_hours", train_cfg.get("sepsis_horizon_hours", 0)))
    patient_agg = str(eval_cfg.get("sepsis_patient_aggregation", "max")).lower()
    if patient_agg not in {"max", "mean", "last"}:
        raise ValueError(f"eval.sepsis_patient_aggregation must be one of ['max','mean','last'], got: {patient_agg}")

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            out = model(batch["x"], batch["mask"])

            combo_idx = CausalTransformer.action_to_combo_index(batch["actions"])
            factual_prob = out["outcome_prob_all"].gather(-1, combo_idx.unsqueeze(-1)).squeeze(-1)
            sepsis_prob = out["sepsis_prob"] if has_sepsis_head else factual_prob
            sepsis_target = build_temporal_target_torch(
                sepsis_label=batch["sepsis_label"],
                mask=batch["mask"],
                mode=sepsis_target_mode,
                horizon_hours=sepsis_horizon_hours,
            )

            valid = batch["mask"]
            y_true.append(batch["y"][valid].cpu().numpy())
            y_prob.append(factual_prob[valid].cpu().numpy())
            y_sepsis_true.append(sepsis_target[valid].cpu().numpy())
            y_sepsis_prob.append(sepsis_prob[valid].cpu().numpy())
            pred_all.append(out["outcome_prob_all"][valid].cpu().numpy())
            true_all.append(batch["y_all"][valid].cpu().numpy())

            pid = batch["patient_id"].cpu().numpy()
            valid_np = valid.cpu().numpy()
            st_np = sepsis_target.cpu().numpy()
            sp_np = sepsis_prob.cpu().numpy()
            for i in range(len(pid)):
                m = valid_np[i]
                if not m.any():
                    continue
                true_seq = st_np[i][m]
                prob_seq = sp_np[i][m]
                p_true = float(np.max(true_seq))
                if patient_agg == "max":
                    p_prob = float(np.max(prob_seq))
                elif patient_agg == "mean":
                    p_prob = float(np.mean(prob_seq))
                else:
                    p_prob = float(prob_seq[-1])
                patient_sepsis_true.append(p_true)
                patient_sepsis_prob.append(p_prob)

    y_true_arr = np.concatenate(y_true) if y_true else np.array([])
    y_prob_arr = np.concatenate(y_prob) if y_prob else np.array([])
    y_sepsis_true_arr = np.concatenate(y_sepsis_true) if y_sepsis_true else np.array([])
    y_sepsis_prob_arr = np.concatenate(y_sepsis_prob) if y_sepsis_prob else np.array([])
    pred_all_arr = np.concatenate(pred_all) if pred_all else np.array([])
    true_all_arr = np.concatenate(true_all) if true_all else np.array([])

    factual = safe_classification_metrics(y_true_arr, y_prob_arr, threshold=threshold) if y_true_arr.size else {}
    sepsis = (
        safe_classification_metrics(y_sepsis_true_arr, y_sepsis_prob_arr, threshold=sepsis_threshold)
        if y_sepsis_true_arr.size
        else {}
    )
    p_true_arr = np.asarray(patient_sepsis_true, dtype=np.float64)
    p_prob_arr = np.asarray(patient_sepsis_prob, dtype=np.float64)
    patient_sepsis = (
        safe_classification_metrics(p_true_arr, p_prob_arr, threshold=sepsis_patient_threshold)
        if p_true_arr.size
        else {}
    )
    factual_error = probability_error_metrics(y_true_arr, y_prob_arr) if y_true_arr.size else {}
    sepsis_error = (
        probability_error_metrics(y_sepsis_true_arr, y_sepsis_prob_arr) if y_sepsis_true_arr.size else {}
    )
    patient_sepsis_error = probability_error_metrics(p_true_arr, p_prob_arr) if p_true_arr.size else {}
    causal = treatment_effect_metrics(pred_all_arr, true_all_arr) if pred_all_arr.size else {}

    payload = {
        "checkpoint": str(model_path),
        "n_test_sequences": len(ds_test),
        "n_test_timesteps": int(y_true_arr.size),
        "classification_threshold": threshold,
        "sepsis_classification_threshold": sepsis_threshold,
        "sepsis_target_mode": sepsis_target_mode,
        "sepsis_horizon_hours": sepsis_horizon_hours,
        "sepsis_patient_aggregation": patient_agg,
        "sepsis_patient_classification_threshold": sepsis_patient_threshold,
        "factual_metrics": factual,
        "sepsis_metrics": sepsis,
        "patient_sepsis_metrics": patient_sepsis,
        "factual_error_metrics": factual_error,
        "sepsis_error_metrics": sepsis_error,
        "patient_sepsis_error_metrics": patient_sepsis_error,
        "treatment_effect_metrics": causal,
    }

    out_eval = out_dir / "eval" / "metrics.json"
    save_json(payload, out_eval)
    return out_eval
