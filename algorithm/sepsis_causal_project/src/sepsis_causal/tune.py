from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from optuna.samplers import GridSampler, TPESampler
from torch.utils.data import DataLoader

from .data import PatientSequenceDataset, collate_patient_batch
from .metrics import probability_error_metrics, safe_classification_metrics, treatment_effect_metrics
from .model import CausalTransformer
from .targets import build_temporal_target_torch
from .train import run_train_with_paths
from .utils import save_json


def _set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    root, leaf = key.split(".", 1)
    cfg[root][leaf] = value


def _sample_bayes(
    trial: optuna.Trial,
    search_space: dict[str, Any],
    base_train_cfg: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    hidden = trial.suggest_categorical(
        "model.hidden_size",
        search_space.get("model.hidden_size", [64, 96, 128, 160, 192]),
    )
    heads = trial.suggest_categorical(
        "model.num_heads",
        search_space.get("model.num_heads", [4, 8]),
    )
    if hidden % heads != 0:
        raise optuna.TrialPruned()
    ff_mult = trial.suggest_categorical(
        "model.ff_multiplier",
        search_space.get("model.ff_multiplier", [2, 3, 4]),
    )

    params["model.hidden_size"] = int(hidden)
    params["model.num_heads"] = int(heads)
    params["model.ff_dim"] = int(hidden * ff_mult)
    params["model.num_layers"] = int(
        trial.suggest_categorical(
            "model.num_layers",
            search_space.get("model.num_layers", [2, 3, 4]),
        )
    )
    params["model.dropout"] = float(
        trial.suggest_float(
            "model.dropout",
            float(search_space.get("model.dropout_min", 0.05)),
            float(search_space.get("model.dropout_max", 0.30)),
        )
    )
    params["train.learning_rate"] = float(
        trial.suggest_float(
            "train.learning_rate",
            float(search_space.get("train.learning_rate_min", 3e-5)),
            float(search_space.get("train.learning_rate_max", 5e-4)),
            log=True,
        )
    )
    params["train.weight_decay"] = float(
        trial.suggest_float(
            "train.weight_decay",
            float(search_space.get("train.weight_decay_min", 1e-6)),
            float(search_space.get("train.weight_decay_max", 5e-2)),
            log=True,
        )
    )
    params["train.batch_size"] = int(
        trial.suggest_categorical(
            "train.batch_size",
            search_space.get("train.batch_size", [32, 64]),
        )
    )
    params["train.lambda_propensity"] = float(
        trial.suggest_float(
            "train.lambda_propensity",
            float(search_space.get("train.lambda_propensity_min", 0.5)),
            float(search_space.get("train.lambda_propensity_max", 2.0)),
        )
    )
    params["train.lambda_balance"] = float(
        trial.suggest_float(
            "train.lambda_balance",
            float(search_space.get("train.lambda_balance_min", 0.01)),
            float(search_space.get("train.lambda_balance_max", 0.6)),
            log=True,
        )
    )
    params["train.lambda_smooth"] = float(
        trial.suggest_float(
            "train.lambda_smooth",
            float(search_space.get("train.lambda_smooth_min", 0.005)),
            float(search_space.get("train.lambda_smooth_max", 0.25)),
            log=True,
        )
    )
    params["train.grad_clip"] = float(
        trial.suggest_float(
            "train.grad_clip",
            float(search_space.get("train.grad_clip_min", 0.5)),
            float(search_space.get("train.grad_clip_max", 2.0)),
        )
    )
    params["train.lambda_sepsis"] = float(
        trial.suggest_float(
            "train.lambda_sepsis",
            float(search_space.get("train.lambda_sepsis_min", 0.5)),
            float(search_space.get("train.lambda_sepsis_max", 3.0)),
        )
    )
    params["train.sepsis_pos_weight"] = float(
        trial.suggest_float(
            "train.sepsis_pos_weight",
            float(search_space.get("train.sepsis_pos_weight_min", 1.0)),
            float(search_space.get("train.sepsis_pos_weight_max", 8.0)),
        )
    )
    sampler_mode = str(base_train_cfg.get("sampler_mode", "none")).lower()
    if sampler_mode == "balanced_patient":
        params["train.sampler_positive_fraction"] = float(
            trial.suggest_float(
                "train.sampler_positive_fraction",
                float(search_space.get("train.sampler_positive_fraction_min", 0.25)),
                float(search_space.get("train.sampler_positive_fraction_max", 0.70)),
            )
        )
    return params


def _sample_grid(trial: optuna.Trial, grid_space: dict[str, list[Any]]) -> dict[str, Any]:
    p: dict[str, Any] = {}
    for k, vals in grid_space.items():
        p[k] = trial.suggest_categorical(k, vals)
    return {
        "model.hidden_size": int(p["model.hidden_size"]),
        "model.num_heads": int(p["model.num_heads"]),
        "model.ff_dim": int(p["model.ff_dim"]),
        "model.num_layers": int(p["model.num_layers"]),
        "model.dropout": float(p["model.dropout"]),
        "train.learning_rate": float(p["train.learning_rate"]),
        "train.weight_decay": float(p["train.weight_decay"]),
        "train.batch_size": int(p["train.batch_size"]),
        "train.lambda_propensity": float(p["train.lambda_propensity"]),
        "train.lambda_balance": float(p["train.lambda_balance"]),
        "train.lambda_smooth": float(p["train.lambda_smooth"]),
        "train.grad_clip": float(p.get("train.grad_clip", 1.0)),
        "train.lambda_sepsis": float(p.get("train.lambda_sepsis", 1.0)),
        "train.sepsis_pos_weight": float(p.get("train.sepsis_pos_weight", 1.0)),
        "train.sampler_positive_fraction": float(p.get("train.sampler_positive_fraction", 0.5)),
    }


def _default_grid_space() -> dict[str, list[Any]]:
    return {
        "model.hidden_size": [96, 128],
        "model.num_heads": [4, 8],
        "model.ff_dim": [256, 384],
        "model.num_layers": [2, 3],
        "model.dropout": [0.1, 0.2],
        "train.learning_rate": [1e-4, 2e-4],
        "train.weight_decay": [1e-4, 1e-3],
        "train.batch_size": [32, 64],
        "train.lambda_propensity": [0.75, 1.0],
        "train.lambda_balance": [0.05, 0.1],
        "train.lambda_smooth": [0.03, 0.05],
        "train.grad_clip": [0.8, 1.2],
        "train.lambda_sepsis": [1.0, 2.0],
        "train.sepsis_pos_weight": [1.5, 3.0, 5.0],
        "train.sampler_positive_fraction": [0.3, 0.4, 0.5, 0.6],
    }


def _load_trial_score(best_model_path: Path) -> tuple[float, int]:
    ckpt = torch.load(best_model_path, map_location="cpu")
    return float(ckpt["best_val_total"]), int(ckpt["epoch"])


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


def _score_checkpoint_on_val(
    best_model_path: Path,
    val_csv: Path,
    batch_size: int,
    num_workers: int,
    factual_threshold: float,
    sepsis_threshold: float,
    sepsis_target_mode: str,
    sepsis_horizon_hours: int,
    sepsis_patient_threshold: float,
    sepsis_patient_aggregation: str,
) -> dict[str, dict[str, float]]:
    ckpt = torch.load(best_model_path, map_location="cpu")
    model = _build_model_from_checkpoint(ckpt)
    has_sepsis_head = (
        "sepsis_head.weight" in ckpt["model_state_dict"]
        and "sepsis_head.bias" in ckpt["model_state_dict"]
    )

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

    y_true = []
    y_prob = []
    y_sepsis_true = []
    y_sepsis_prob = []
    p_sepsis_true = []
    p_sepsis_prob = []
    pred_all = []
    true_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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

            valid_np = valid.cpu().numpy()
            st_np = sepsis_target.cpu().numpy()
            sp_np = sepsis_prob.cpu().numpy()
            for i in range(valid_np.shape[0]):
                m = valid_np[i]
                if not m.any():
                    continue
                true_seq = st_np[i][m]
                prob_seq = sp_np[i][m]
                p_true = float(np.max(true_seq))
                if sepsis_patient_aggregation == "max":
                    p_prob = float(np.max(prob_seq))
                elif sepsis_patient_aggregation == "mean":
                    p_prob = float(np.mean(prob_seq))
                elif sepsis_patient_aggregation == "last":
                    p_prob = float(prob_seq[-1])
                else:
                    raise ValueError(
                        "sepsis_patient_aggregation must be one of ['max','mean','last'], "
                        f"got: {sepsis_patient_aggregation}"
                    )
                p_sepsis_true.append(p_true)
                p_sepsis_prob.append(p_prob)

    y_true_arr = np.concatenate(y_true) if y_true else np.array([])
    y_prob_arr = np.concatenate(y_prob) if y_prob else np.array([])
    y_sepsis_true_arr = np.concatenate(y_sepsis_true) if y_sepsis_true else np.array([])
    y_sepsis_prob_arr = np.concatenate(y_sepsis_prob) if y_sepsis_prob else np.array([])
    p_sepsis_true_arr = np.asarray(p_sepsis_true, dtype=np.float64)
    p_sepsis_prob_arr = np.asarray(p_sepsis_prob, dtype=np.float64)
    pred_all_arr = np.concatenate(pred_all) if pred_all else np.array([])
    true_all_arr = np.concatenate(true_all) if true_all else np.array([])

    factual = (
        safe_classification_metrics(y_true_arr, y_prob_arr, threshold=factual_threshold)
        if y_true_arr.size
        else {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
    )
    sepsis = (
        safe_classification_metrics(y_sepsis_true_arr, y_sepsis_prob_arr, threshold=sepsis_threshold)
        if y_sepsis_true_arr.size
        else {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
    )
    sepsis_patient = (
        safe_classification_metrics(
            p_sepsis_true_arr,
            p_sepsis_prob_arr,
            threshold=sepsis_patient_threshold,
        )
        if p_sepsis_true_arr.size
        else {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
    )
    factual_error = (
        probability_error_metrics(y_true_arr, y_prob_arr)
        if y_true_arr.size
        else {"mae": float("nan"), "rmse": float("nan")}
    )
    sepsis_error = (
        probability_error_metrics(y_sepsis_true_arr, y_sepsis_prob_arr)
        if y_sepsis_true_arr.size
        else {"mae": float("nan"), "rmse": float("nan")}
    )
    sepsis_patient_error = (
        probability_error_metrics(p_sepsis_true_arr, p_sepsis_prob_arr)
        if p_sepsis_true_arr.size
        else {"mae": float("nan"), "rmse": float("nan")}
    )
    causal = (
        treatment_effect_metrics(pred_all_arr, true_all_arr)
        if pred_all_arr.size
        else {"pehe": float("nan"), "ate_error": float("nan"), "policy_regret": float("nan")}
    )
    return {
        "factual": factual,
        "sepsis": sepsis,
        "sepsis_patient": sepsis_patient,
        "factual_error": factual_error,
        "sepsis_error": sepsis_error,
        "sepsis_patient_error": sepsis_patient_error,
        "causal": causal,
    }


def _objective_from_metrics(
    tune_cfg: dict[str, Any],
    best_val_total: float,
    metric_bundle: dict[str, dict[str, float]],
) -> tuple[float, str]:
    objective_metric = str(tune_cfg.get("objective_metric", "val_total")).lower()
    factual = metric_bundle["factual"]
    classification_level = str(tune_cfg.get("classification_level", "patient")).lower()
    if classification_level not in {"timestep", "patient"}:
        raise ValueError(
            f"tune.classification_level must be one of ['timestep','patient'], got: {classification_level}"
        )
    sepsis = metric_bundle["sepsis"] if classification_level == "timestep" else metric_bundle["sepsis_patient"]
    causal = metric_bundle["causal"]
    sepsis_error = (
        metric_bundle["sepsis_error"]
        if classification_level == "timestep"
        else metric_bundle["sepsis_patient_error"]
    )

    if objective_metric in {"val_total", "total_loss", "loss"}:
        return float(best_val_total), objective_metric
    if objective_metric in {"pehe"}:
        return float(causal["pehe"]), objective_metric
    if objective_metric in {"ate_error", "ate"}:
        return float(causal["ate_error"]), objective_metric
    if objective_metric in {"policy_regret", "regret"}:
        return float(causal["policy_regret"]), objective_metric
    if objective_metric in {"negative_auroc", "neg_auroc", "max_auroc"}:
        auroc = float(factual.get("auroc", float("nan")))
        if np.isnan(auroc):
            auroc = 0.5
        return float(-auroc), objective_metric
    if objective_metric in {"negative_sepsis_auroc", "neg_sepsis_auroc", "max_sepsis_auroc"}:
        auroc = float(sepsis.get("auroc", float("nan")))
        if np.isnan(auroc):
            auroc = 0.5
        return float(-auroc), objective_metric
    if objective_metric in {"negative_patient_sepsis_auroc", "neg_patient_sepsis_auroc"}:
        auroc = float(metric_bundle["sepsis_patient"].get("auroc", float("nan")))
        if np.isnan(auroc):
            auroc = 0.5
        return float(-auroc), objective_metric
    if objective_metric in {"negative_patient_sepsis_auprc", "neg_patient_sepsis_auprc"}:
        auprc = float(metric_bundle["sepsis_patient"].get("auprc", float("nan")))
        if np.isnan(auprc):
            auprc = 0.0
        return float(-auprc), objective_metric
    if objective_metric in {"negative_patient_sepsis_f1", "neg_patient_sepsis_f1"}:
        f1 = float(metric_bundle["sepsis_patient"].get("f1", float("nan")))
        if np.isnan(f1):
            f1 = 0.0
        return float(-f1), objective_metric
    if objective_metric in {"sepsis_rmse"}:
        return float(sepsis_error["rmse"]), objective_metric
    if objective_metric in {"sepsis_mae"}:
        return float(sepsis_error["mae"]), objective_metric
    if objective_metric in {"causal_combo", "causal_weighted", "treatment_combo"}:
        weights = tune_cfg.get("causal_objective_weights", {})
        w_pehe = float(weights.get("pehe", 1.0))
        w_ate = float(weights.get("ate_error", 1.0))
        w_regret = float(weights.get("policy_regret", 1.5))
        score = (
            w_pehe * float(causal["pehe"])
            + w_ate * float(causal["ate_error"])
            + w_regret * float(causal["policy_regret"])
        )
        return float(score), objective_metric
    if objective_metric in {"hybrid_combo", "hybrid_treatment_sepsis"}:
        weights = tune_cfg.get("hybrid_objective_weights", {})
        w_pehe = float(weights.get("pehe", 1.0))
        w_ate = float(weights.get("ate_error", 1.0))
        w_regret = float(weights.get("policy_regret", 1.5))
        w_one_minus_auroc = float(weights.get("one_minus_sepsis_auroc", 1.0))
        w_rmse = float(weights.get("sepsis_rmse", 1.0))
        sepsis_auroc = float(sepsis.get("auroc", float("nan")))
        if np.isnan(sepsis_auroc):
            sepsis_auroc = 0.5
        score = (
            w_pehe * float(causal["pehe"])
            + w_ate * float(causal["ate_error"])
            + w_regret * float(causal["policy_regret"])
            + w_one_minus_auroc * float(1.0 - sepsis_auroc)
            + w_rmse * float(sepsis_error["rmse"])
        )
        return float(score), objective_metric

    valid = [
        "val_total",
        "pehe",
        "ate_error",
        "policy_regret",
        "causal_combo",
        "negative_auroc",
        "negative_sepsis_auroc",
        "negative_patient_sepsis_auroc",
        "negative_patient_sepsis_auprc",
        "negative_patient_sepsis_f1",
        "sepsis_rmse",
        "sepsis_mae",
        "hybrid_combo",
    ]
    raise ValueError(f"Unknown tune.objective_metric={objective_metric}. Choose one of: {valid}")


def run_tuning(config: dict[str, Any], out_dir: Path) -> Path:
    tune_cfg = config.get("tune", {})
    if not tune_cfg:
        raise ValueError("Missing `tune` section in config.")

    prepared_dir = out_dir / "prepared"
    if not (prepared_dir / "train.csv").exists():
        raise FileNotFoundError("Prepared data missing. Run prepare first.")

    method = str(tune_cfg.get("method", "bayes")).lower()
    n_trials = int(tune_cfg.get("n_trials", 20))
    max_total_trials = tune_cfg.get("max_total_trials")
    timeout_sec = tune_cfg.get("timeout_sec")
    seed = int(config["seed"])
    direction = str(tune_cfg.get("direction", "minimize"))
    search_space = tune_cfg.get("search_space", {})
    objective_metric = str(tune_cfg.get("objective_metric", "val_total")).lower()

    tune_root = out_dir / "tuning"
    tune_root.mkdir(parents=True, exist_ok=True)
    study_name = str(tune_cfg.get("study_name", "sepsis_hparam_tuning"))
    study_dir = tune_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = str(study_dir / "optuna_study.db")
    storage_uri = f"sqlite:///{storage.replace('\\', '/')}"

    grid_space: dict[str, list[Any]] | None = None
    if method == "bayes":
        sampler = TPESampler(seed=seed)
    elif method == "grid":
        grid_space = tune_cfg.get("grid_space", _default_grid_space())
        sampler = GridSampler(grid_space)
        total_points = 1
        for vals in grid_space.values():
            total_points *= len(vals)
        n_trials = int(n_trials or total_points)
    else:
        raise ValueError(f"Unknown tuning method: {method}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

    trials_before = len(study.trials)
    if max_total_trials is None:
        n_trials_to_run = n_trials
    else:
        target_total = int(max_total_trials)
        n_trials_to_run = max(0, target_total - trials_before)

    def objective(trial: optuna.Trial) -> float:
        params = (
            _sample_bayes(
                trial,
                search_space,
                base_train_cfg=config.get("train", {}),
            )
            if method == "bayes"
            else _sample_grid(trial, grid_space=grid_space or {})
        )
        if params["model.hidden_size"] % params["model.num_heads"] != 0:
            raise optuna.TrialPruned()

        trial_cfg = copy.deepcopy(config)
        for k, v in params.items():
            _set_nested(trial_cfg, k, v)

        # Faster objective while preserving ranking signal.
        trial_cfg["train"]["epochs"] = int(tune_cfg.get("trial_epochs", 6))
        trial_cfg["train"]["early_stopping_patience"] = int(tune_cfg.get("trial_patience", 2))
        trial_cfg["train"]["max_train_patients"] = tune_cfg.get("max_train_patients", 5000)
        trial_cfg["train"]["max_val_patients"] = tune_cfg.get("max_val_patients", 1500)

        trial_out = study_dir / f"trial_{trial.number:04d}"
        trial_model_dir = trial_out / "model"
        best_model_path = run_train_with_paths(
            config=trial_cfg,
            out_dir=trial_out,
            prepared_dir=prepared_dir,
            model_dir=trial_model_dir,
        )
        best_val, best_epoch = _load_trial_score(best_model_path)
        trial_eval_cfg = trial_cfg.get("eval", {})
        trial_train_cfg = trial_cfg.get("train", {})
        sepsis_target_mode = str(
            trial_eval_cfg.get("sepsis_target_mode", trial_train_cfg.get("sepsis_target_mode", "current"))
        ).lower()
        sepsis_horizon_hours = int(
            trial_eval_cfg.get("sepsis_horizon_hours", trial_train_cfg.get("sepsis_horizon_hours", 0))
        )
        sepsis_patient_aggregation = str(trial_eval_cfg.get("sepsis_patient_aggregation", "max")).lower()
        sepsis_threshold = float(
            trial_eval_cfg.get(
                "sepsis_threshold",
                trial_eval_cfg.get("threshold", 0.5),
            )
        )
        sepsis_patient_threshold = float(
            trial_eval_cfg.get("sepsis_patient_threshold", sepsis_threshold)
        )
        metric_bundle = _score_checkpoint_on_val(
            best_model_path=best_model_path,
            val_csv=prepared_dir / "val.csv",
            batch_size=int(trial_cfg["train"]["batch_size"]),
            num_workers=int(trial_cfg["train"].get("num_workers", 0)),
            factual_threshold=float(trial_cfg.get("eval", {}).get("threshold", 0.5)),
            sepsis_threshold=sepsis_threshold,
            sepsis_target_mode=sepsis_target_mode,
            sepsis_horizon_hours=sepsis_horizon_hours,
            sepsis_patient_threshold=sepsis_patient_threshold,
            sepsis_patient_aggregation=sepsis_patient_aggregation,
        )
        objective_value, objective_name = _objective_from_metrics(
            tune_cfg=tune_cfg,
            best_val_total=best_val,
            metric_bundle=metric_bundle,
        )

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_model_path", str(best_model_path))
        trial.set_user_attr("best_val_total", best_val)
        trial.set_user_attr("objective_metric", objective_name)
        trial.set_user_attr("objective_value", float(objective_value))
        trial.set_user_attr("val_factual_auroc", float(metric_bundle["factual"]["auroc"]))
        trial.set_user_attr("val_factual_auprc", float(metric_bundle["factual"]["auprc"]))
        trial.set_user_attr("val_factual_f1", float(metric_bundle["factual"]["f1"]))
        trial.set_user_attr("val_sepsis_auroc", float(metric_bundle["sepsis"]["auroc"]))
        trial.set_user_attr("val_sepsis_auprc", float(metric_bundle["sepsis"]["auprc"]))
        trial.set_user_attr("val_sepsis_f1", float(metric_bundle["sepsis"]["f1"]))
        trial.set_user_attr("val_sepsis_mae", float(metric_bundle["sepsis_error"]["mae"]))
        trial.set_user_attr("val_sepsis_rmse", float(metric_bundle["sepsis_error"]["rmse"]))
        trial.set_user_attr("val_patient_sepsis_auroc", float(metric_bundle["sepsis_patient"]["auroc"]))
        trial.set_user_attr("val_patient_sepsis_auprc", float(metric_bundle["sepsis_patient"]["auprc"]))
        trial.set_user_attr("val_patient_sepsis_f1", float(metric_bundle["sepsis_patient"]["f1"]))
        trial.set_user_attr("val_patient_sepsis_mae", float(metric_bundle["sepsis_patient_error"]["mae"]))
        trial.set_user_attr("val_patient_sepsis_rmse", float(metric_bundle["sepsis_patient_error"]["rmse"]))
        trial.set_user_attr("val_pehe", float(metric_bundle["causal"]["pehe"]))
        trial.set_user_attr("val_ate_error", float(metric_bundle["causal"]["ate_error"]))
        trial.set_user_attr("val_policy_regret", float(metric_bundle["causal"]["policy_regret"]))
        return float(objective_value)

    if n_trials_to_run > 0:
        study.optimize(objective, n_trials=n_trials_to_run, timeout=timeout_sec, gc_after_trial=True)
    elif not study.trials:
        raise ValueError(
            "No trials scheduled and no existing study trials found. "
            "Increase `tune.n_trials` or `tune.max_total_trials`."
        )

    result = {
        "method": method,
        "direction": direction,
        "objective_metric": objective_metric,
        "classification_level": tune_cfg.get("classification_level", "patient"),
        "causal_objective_weights": tune_cfg.get("causal_objective_weights", None),
        "hybrid_objective_weights": tune_cfg.get("hybrid_objective_weights", None),
        "n_trials_requested": n_trials,
        "n_trials_requested_this_run": n_trials_to_run,
        "n_trials_completed_before": trials_before,
        "n_trials_completed": len(study.trials),
        "n_trials_target_total": (None if max_total_trials is None else int(max_total_trials)),
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "storage": storage_uri,
    }
    save_json(result, study_dir / "best_result.json")
    save_json(
        {
            "trials": [
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": (None if t.value is None else float(t.value)),
                    "params": t.params,
                    "user_attrs": t.user_attrs,
                }
                for t in study.trials
            ]
        },
        study_dir / "trials.json",
    )

    # Render best hyperparameters as trainable config patch.
    best_cfg_patch: dict[str, Any] = {"model": {}, "train": {}}
    for k, v in study.best_params.items():
        if k == "model.ff_multiplier":
            continue
        root, leaf = k.split(".", 1)
        best_cfg_patch[root][leaf] = v
    if "model.ff_multiplier" in study.best_params:
        hidden = int(best_cfg_patch["model"]["hidden_size"])
        ff_mult = int(study.best_params["model.ff_multiplier"])
        best_cfg_patch["model"]["ff_dim"] = hidden * ff_mult

    save_json(best_cfg_patch, study_dir / "best_config_patch.json")
    return study_dir / "best_result.json"
