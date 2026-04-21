from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import PatientSequenceDataset, collate_patient_batch
from .model import CausalTransformer
from .prepare import run_prepare
from .targets import build_temporal_target_torch
from .tune import run_tuning
from .utils import save_json


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_model_from_checkpoint(ckpt: dict[str, Any]) -> CausalTransformer:
    mcfg = ckpt["model_config"]
    model = CausalTransformer(
        input_dim=int(mcfg["input_dim"]),
        hidden_size=int(mcfg["hidden_size"]),
        num_layers=int(mcfg["num_layers"]),
        num_heads=int(mcfg["num_heads"]),
        ff_dim=int(mcfg["ff_dim"]),
        dropout=float(mcfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


def _collect_val_probs(
    model_path: Path,
    val_csv: Path,
    batch_size: int,
    num_workers: int,
    target: str,
    sepsis_target_mode: str,
    sepsis_horizon_hours: int,
    target_level: str,
    sepsis_patient_aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(model_path, map_location="cpu")
    model = _build_model_from_checkpoint(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds_val = PatientSequenceDataset(val_csv)
    loader = DataLoader(
        ds_val,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_patient_batch,
    )

    has_sepsis_head = (
        "sepsis_head.weight" in ckpt["model_state_dict"]
        and "sepsis_head.bias" in ckpt["model_state_dict"]
    )

    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"], batch["mask"])
            combo_idx = CausalTransformer.action_to_combo_index(batch["actions"])
            factual_prob = out["outcome_prob_all"].gather(-1, combo_idx.unsqueeze(-1)).squeeze(-1)
            sepsis_prob = out["sepsis_prob"] if has_sepsis_head else factual_prob
            valid = batch["mask"]
            if target == "sepsis":
                sepsis_target = build_temporal_target_torch(
                    sepsis_label=batch["sepsis_label"],
                    mask=valid,
                    mode=sepsis_target_mode,
                    horizon_hours=sepsis_horizon_hours,
                )
                if target_level == "patient":
                    valid_np = valid.cpu().numpy()
                    st_np = sepsis_target.cpu().numpy()
                    sp_np = sepsis_prob.cpu().numpy()
                    for i in range(valid_np.shape[0]):
                        m = valid_np[i]
                        if not m.any():
                            continue
                        ts = st_np[i][m]
                        ps = sp_np[i][m]
                        y_true.append(np.array([float(np.max(ts))], dtype=np.float64))
                        if sepsis_patient_aggregation == "max":
                            y_prob.append(np.array([float(np.max(ps))], dtype=np.float64))
                        elif sepsis_patient_aggregation == "mean":
                            y_prob.append(np.array([float(np.mean(ps))], dtype=np.float64))
                        elif sepsis_patient_aggregation == "last":
                            y_prob.append(np.array([float(ps[-1])], dtype=np.float64))
                        else:
                            raise ValueError(
                                "optimize.sepsis_patient_aggregation must be one of ['max','mean','last'], "
                                f"got: {sepsis_patient_aggregation}"
                            )
                else:
                    y_true.append(sepsis_target[valid].cpu().numpy())
                    y_prob.append(sepsis_prob[valid].cpu().numpy())
            else:
                y_true.append(batch["y"][valid].cpu().numpy())
                y_prob.append(factual_prob[valid].cpu().numpy())

    y_true_arr = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_prob_arr = np.concatenate(y_prob) if y_prob else np.array([], dtype=np.float64)
    return y_true_arr, y_prob_arr


def _calibrate_threshold(
    model_path: Path,
    prepared_dir: Path,
    batch_size: int,
    num_workers: int,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    target: str,
    sepsis_target_mode: str,
    sepsis_horizon_hours: int,
    target_level: str,
    sepsis_patient_aggregation: str,
) -> dict[str, Any]:
    val_csv = prepared_dir / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing validation split for threshold calibration: {val_csv}")

    y_true, y_prob = _collect_val_probs(
        model_path=model_path,
        val_csv=val_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        target=target,
        sepsis_target_mode=sepsis_target_mode,
        sepsis_horizon_hours=sepsis_horizon_hours,
        target_level=target_level,
        sepsis_patient_aggregation=sepsis_patient_aggregation,
    )
    if y_true.size == 0:
        raise ValueError("No validation samples found for threshold calibration.")

    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    f1_scores = np.array(
        [
            f1_score(y_true, (y_prob >= th).astype(np.int64), zero_division=0)
            for th in thresholds
        ],
        dtype=np.float64,
    )
    max_f1 = float(np.max(f1_scores))
    best_indices = np.flatnonzero(np.isclose(f1_scores, max_f1))
    # Stable tie-break: prefer threshold closest to 0.5.
    best_idx = int(best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))])

    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "threshold_min": float(threshold_min),
        "threshold_max": float(threshold_max),
        "threshold_steps": int(threshold_steps),
        "threshold_target": target,
        "threshold_level": target_level,
        "n_val_timesteps": int(y_true.size),
        "positive_rate": float(np.mean(y_true)),
    }


def _merge_best_patch(base_config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    for section, values in patch.items():
        if not isinstance(values, dict):
            continue
        if section not in cfg or not isinstance(cfg[section], dict):
            cfg[section] = {}
        cfg[section].update(values)
    return cfg


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_optimize(config: dict[str, Any], data_root: Path, out_dir: Path) -> Path:
    optimize_cfg = config.get("optimize", {})
    auto_prepare = bool(optimize_cfg.get("auto_prepare", True))

    prepared_dir = out_dir / "prepared"
    if not (prepared_dir / "train.csv").exists():
        if not auto_prepare:
            raise FileNotFoundError(
                "Prepared data missing and optimize.auto_prepare=false. "
                "Run prepare first or set optimize.auto_prepare=true."
            )
        run_prepare(config=config, data_root=data_root, out_dir=out_dir)

    tune_result_path = run_tuning(config=config, out_dir=out_dir)
    tune_result = _load_json(tune_result_path)
    study_dir = tune_result_path.parent
    patch_path = study_dir / "best_config_patch.json"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing best patch from tuning: {patch_path}")

    best_patch = _load_json(patch_path)
    optimized_config = _merge_best_patch(config, best_patch)

    if bool(optimize_cfg.get("reset_train_patient_limits", True)):
        optimized_config.setdefault("train", {})
        optimized_config["train"]["max_train_patients"] = None
        optimized_config["train"]["max_val_patients"] = None

    final_out_dir = optimize_cfg.get("final_out_dir")
    if final_out_dir is None:
        final_out_dir = str((out_dir.parent / f"{out_dir.name}_final_optimized").as_posix())
    optimized_config.setdefault("paths", {})
    optimized_config["paths"]["out_dir"] = str(final_out_dir)

    threshold_result: dict[str, Any] | None = None
    if bool(optimize_cfg.get("calibrate_threshold", True)):
        threshold_target = str(optimize_cfg.get("threshold_target", "sepsis")).lower()
        if threshold_target not in {"sepsis", "factual"}:
            raise ValueError(
                f"optimize.threshold_target must be 'sepsis' or 'factual', got: {threshold_target}"
            )
        threshold_level = str(optimize_cfg.get("threshold_level", "timestep")).lower()
        if threshold_level not in {"timestep", "patient"}:
            raise ValueError(
                f"optimize.threshold_level must be 'timestep' or 'patient', got: {threshold_level}"
            )
        sepsis_target_mode = str(
            optimize_cfg.get(
                "sepsis_target_mode",
                optimized_config.get("eval", {}).get(
                    "sepsis_target_mode",
                    optimized_config.get("train", {}).get("sepsis_target_mode", "current"),
                ),
            )
        ).lower()
        sepsis_horizon_hours = int(
            optimize_cfg.get(
                "sepsis_horizon_hours",
                optimized_config.get("eval", {}).get(
                    "sepsis_horizon_hours",
                    optimized_config.get("train", {}).get("sepsis_horizon_hours", 0),
                ),
            )
        )
        sepsis_patient_aggregation = str(
            optimize_cfg.get(
                "sepsis_patient_aggregation",
                optimized_config.get("eval", {}).get("sepsis_patient_aggregation", "max"),
            )
        ).lower()
        best_model_path_raw = tune_result.get("best_user_attrs", {}).get("best_model_path")
        if best_model_path_raw is None:
            raise ValueError("Tuning result missing best model path for threshold calibration.")
        best_model_path = Path(best_model_path_raw)
        threshold_result = _calibrate_threshold(
            model_path=best_model_path,
            prepared_dir=prepared_dir,
            batch_size=int(optimized_config["train"]["batch_size"]),
            num_workers=int(optimized_config["train"].get("num_workers", 0)),
            threshold_min=float(optimize_cfg.get("threshold_min", 0.05)),
            threshold_max=float(optimize_cfg.get("threshold_max", 0.95)),
            threshold_steps=int(optimize_cfg.get("threshold_steps", 91)),
            target=threshold_target,
            sepsis_target_mode=sepsis_target_mode,
            sepsis_horizon_hours=sepsis_horizon_hours,
            target_level=threshold_level,
            sepsis_patient_aggregation=sepsis_patient_aggregation,
        )
        optimized_config.setdefault("eval", {})
        if threshold_target == "sepsis" and threshold_level == "patient":
            optimized_config["eval"]["sepsis_patient_threshold"] = threshold_result["best_threshold"]
        elif threshold_target == "sepsis":
            optimized_config["eval"]["sepsis_threshold"] = threshold_result["best_threshold"]
        else:
            optimized_config["eval"]["threshold"] = threshold_result["best_threshold"]
        save_json(threshold_result, study_dir / "best_threshold.json")

    default_cfg_name = f"optimized_{study_dir.name}.yaml"
    output_cfg_raw = optimize_cfg.get("output_config_path", f"configs/final/{default_cfg_name}")
    output_cfg_path = _resolve_project_path(output_cfg_raw)
    _write_yaml(optimized_config, output_cfg_path)

    summary = {
        "tuning_result_path": str(tune_result_path),
        "best_patch_path": str(patch_path),
        "best_trial_number": int(tune_result["best_trial_number"]),
        "best_objective_value": float(tune_result["best_value"]),
        "threshold_calibration": threshold_result,
        "optimized_config_path": str(output_cfg_path),
        "recommended_training_command": (
            f"python -m sepsis_causal.cli --config {output_cfg_path.as_posix()} train"
        ),
        "recommended_evaluation_command": (
            f"python -m sepsis_causal.cli --config {output_cfg_path.as_posix()} evaluate"
        ),
    }
    save_json(summary, study_dir / "optimization_summary.json")
    return output_cfg_path
