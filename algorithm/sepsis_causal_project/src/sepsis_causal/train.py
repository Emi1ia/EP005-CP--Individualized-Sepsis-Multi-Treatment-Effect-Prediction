from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import PatientSequenceDataset, collate_patient_batch
from .metrics import probability_error_metrics, safe_classification_metrics
from .model import build_model_from_config, compute_losses, normalize_model_config
from .targets import build_temporal_target_torch


def _normalize_progress_checkpoints(raw: Any) -> dict[int, dict[str, dict[str, float]]]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError("train.progress_checkpoints must be a list of checkpoint rules")

    out: dict[int, dict[str, dict[str, float]]] = {}
    for i, rule in enumerate(raw):
        if not isinstance(rule, dict):
            raise ValueError(f"train.progress_checkpoints[{i}] must be a dict")
        if "epoch" not in rule:
            raise ValueError(f"train.progress_checkpoints[{i}] missing required key: epoch")

        ep = int(rule["epoch"])
        if ep <= 0:
            raise ValueError(f"train.progress_checkpoints[{i}].epoch must be > 0")

        min_rules_raw = rule.get("min", {}) or {}
        max_rules_raw = rule.get("max", {}) or {}
        if not isinstance(min_rules_raw, dict) or not isinstance(max_rules_raw, dict):
            raise ValueError(
                f"train.progress_checkpoints[{i}] min/max must be mapping objects"
            )

        min_rules = {str(k): float(v) for k, v in min_rules_raw.items()}
        max_rules = {str(k): float(v) for k, v in max_rules_raw.items()}
        out[ep] = {"min": min_rules, "max": max_rules}
    return out


def _checkpoint_rule_failure(
    epoch: int,
    val_metrics: dict[str, Any],
    rule: dict[str, dict[str, float]],
) -> str | None:
    for metric_name, threshold in rule.get("min", {}).items():
        if metric_name not in val_metrics:
            raise ValueError(
                f"progress checkpoint epoch={epoch} references unknown val metric: {metric_name}"
            )
        v = float(val_metrics[metric_name])
        if np.isnan(v) or v < float(threshold):
            return f"{metric_name}={v:.4f} < min={float(threshold):.4f}"

    for metric_name, threshold in rule.get("max", {}).items():
        if metric_name not in val_metrics:
            raise ValueError(
                f"progress checkpoint epoch={epoch} references unknown val metric: {metric_name}"
            )
        v = float(val_metrics[metric_name])
        if np.isnan(v) or v > float(threshold):
            return f"{metric_name}={v:.4f} > max={float(threshold):.4f}"

    return None


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _collect_masked(arr: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return arr[mask].detach().cpu().numpy()


def _compute_l1_penalty(model: torch.nn.Module, include_bias: bool) -> torch.Tensor:
    try:
        zero = next(model.parameters()).new_tensor(0.0)
    except StopIteration:  # pragma: no cover
        return torch.tensor(0.0)

    reg = zero
    denom = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if (not include_bias) and p.ndim <= 1:
            continue
        reg = reg + p.abs().sum()
        denom += int(p.numel())
    if denom <= 0:
        return zero
    return reg / float(denom)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_propensity: float,
    lambda_balance: float,
    lambda_smooth: float,
    lambda_sepsis: float,
    sepsis_pos_weight: float,
    l1_weight: float,
    l1_include_bias: bool,
    grad_clip: float,
    metric_target: str,
    metric_threshold: float,
    patient_metric_aggregation: str,
    patient_metric_threshold: float,
    sepsis_target_mode: str,
    sepsis_horizon_hours: int,
    compute_metrics: bool = True,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "total": 0.0,
        "outcome": 0.0,
        "propensity": 0.0,
        "balance": 0.0,
        "smooth": 0.0,
        "sepsis": 0.0,
        "l1": 0.0,
    }
    steps = 0

    y_true_list = []
    y_prob_list = []
    p_true_list = []
    p_prob_list = []

    for batch in loader:
        batch = _move_batch(batch, device)
        sepsis_target = build_temporal_target_torch(
            sepsis_label=batch["sepsis_label"],
            mask=batch["mask"],
            mode=sepsis_target_mode,
            horizon_hours=sepsis_horizon_hours,
        )
        batch["sepsis_target"] = sepsis_target
        outputs = model(batch["x"], batch["mask"])
        loss_dict = compute_losses(
            outputs,
            batch,
            lambda_propensity=lambda_propensity,
            lambda_balance=lambda_balance,
            lambda_smooth=lambda_smooth,
            lambda_sepsis=lambda_sepsis,
            sepsis_pos_weight=sepsis_pos_weight,
        )
        l1_penalty = (
            _compute_l1_penalty(model, include_bias=l1_include_bias)
            if l1_weight > 0
            else torch.tensor(0.0, device=loss_dict["total"].device)
        )
        total_with_reg = loss_dict["total"] + (float(l1_weight) * l1_penalty)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            total_with_reg.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        totals["total"] += float(total_with_reg.detach().cpu())
        totals["outcome"] += float(loss_dict["outcome"].detach().cpu())
        totals["propensity"] += float(loss_dict["propensity"].detach().cpu())
        totals["balance"] += float(loss_dict["balance"].detach().cpu())
        totals["smooth"] += float(loss_dict["smooth"].detach().cpu())
        totals["sepsis"] += float(loss_dict["sepsis"].detach().cpu())
        totals["l1"] += float(l1_penalty.detach().cpu())
        steps += 1

        if compute_metrics:
            if metric_target == "sepsis":
                y_true_list.append(_collect_masked(sepsis_target, batch["mask"]))
                y_prob_list.append(_collect_masked(loss_dict["sepsis_prob"], batch["mask"]))
                valid_np = batch["mask"].detach().cpu().numpy()
                st_np = sepsis_target.detach().cpu().numpy()
                sp_np = loss_dict["sepsis_prob"].detach().cpu().numpy()
                for i in range(valid_np.shape[0]):
                    m = valid_np[i]
                    if not m.any():
                        continue
                    true_seq = st_np[i][m]
                    prob_seq = sp_np[i][m]
                    p_true = float(np.max(true_seq))
                    if patient_metric_aggregation == "max":
                        p_prob = float(np.max(prob_seq))
                    elif patient_metric_aggregation == "mean":
                        p_prob = float(np.mean(prob_seq))
                    elif patient_metric_aggregation == "last":
                        p_prob = float(prob_seq[-1])
                    else:  # pragma: no cover
                        raise ValueError(
                            "train.patient_metric_aggregation must be one of ['max','mean','last'], "
                            f"got: {patient_metric_aggregation}"
                        )
                    p_true_list.append(p_true)
                    p_prob_list.append(p_prob)
            else:
                y_true_list.append(_collect_masked(batch["y"], batch["mask"]))
                y_prob_list.append(_collect_masked(loss_dict["factual_prob"], batch["mask"]))

    agg = {k: (v / max(steps, 1)) for k, v in totals.items()}
    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_prob = np.concatenate(y_prob_list) if y_prob_list else np.array([])
    cls_metrics = (
        safe_classification_metrics(y_true, y_prob, threshold=metric_threshold)
        if (compute_metrics and y_true.size)
        else {"auroc": np.nan, "auprc": np.nan, "f1": np.nan}
    )
    err_metrics = (
        probability_error_metrics(y_true, y_prob)
        if (compute_metrics and y_true.size)
        else {"mae": np.nan, "mse": np.nan, "rmse": np.nan}
    )
    agg.update(cls_metrics)
    agg.update(err_metrics)
    if metric_target == "sepsis":
        p_true = np.asarray(p_true_list, dtype=np.float64)
        p_prob = np.asarray(p_prob_list, dtype=np.float64)
        p_cls_metrics = (
            safe_classification_metrics(p_true, p_prob, threshold=patient_metric_threshold)
            if (compute_metrics and p_true.size)
            else {"auroc": np.nan, "auprc": np.nan, "f1": np.nan}
        )
        p_err_metrics = (
            probability_error_metrics(p_true, p_prob)
            if (compute_metrics and p_true.size)
            else {"mae": np.nan, "mse": np.nan, "rmse": np.nan}
        )
        agg.update(
            {
                "patient_auroc": p_cls_metrics["auroc"],
                "patient_auprc": p_cls_metrics["auprc"],
                "patient_f1": p_cls_metrics["f1"],
                "patient_mae": p_err_metrics["mae"],
                "patient_mse": p_err_metrics["mse"],
                "patient_rmse": p_err_metrics["rmse"],
            }
        )
    return agg


def run_train(config: dict[str, Any], out_dir: Path) -> Path:
    return run_train_with_paths(config=config, out_dir=out_dir, prepared_dir=None, model_dir=None)


def run_train_with_paths(
    config: dict[str, Any],
    out_dir: Path,
    prepared_dir: Path | None = None,
    model_dir: Path | None = None,
) -> Path:
    train_cfg = config["train"]
    model_cfg = config["model"]
    seed = int(config["seed"])
    metric_target = str(train_cfg.get("metric_target", "sepsis")).lower()
    if metric_target not in {"sepsis", "factual"}:
        raise ValueError(f"train.metric_target must be one of ['sepsis', 'factual'], got: {metric_target}")
    metric_threshold = float(train_cfg.get("metric_threshold", config.get("eval", {}).get("threshold", 0.5)))
    patient_metric_aggregation = str(
        train_cfg.get(
            "patient_metric_aggregation",
            config.get("eval", {}).get("sepsis_patient_aggregation", "max"),
        )
    ).lower()
    if patient_metric_aggregation not in {"max", "mean", "last"}:
        raise ValueError(
            "train.patient_metric_aggregation must be one of ['max','mean','last'], "
            f"got: {patient_metric_aggregation}"
        )
    patient_metric_threshold = float(
        train_cfg.get(
            "patient_metric_threshold",
            config.get("eval", {}).get("sepsis_patient_threshold", metric_threshold),
        )
    )
    sepsis_target_mode = str(train_cfg.get("sepsis_target_mode", "current")).lower()
    sepsis_horizon_hours = int(train_cfg.get("sepsis_horizon_hours", 0))
    l1_weight = float(max(0.0, train_cfg.get("l1_weight", 0.0)))
    l1_include_bias = bool(train_cfg.get("l1_include_bias", False))
    compute_train_metrics = bool(train_cfg.get("compute_train_metrics", True))
    early_metric = str(train_cfg.get("early_stopping_metric", "total"))
    early_mode = str(
        train_cfg.get(
            "early_stopping_mode",
            ("min" if early_metric == "total" else "max"),
        )
    ).lower()
    if early_mode not in {"min", "max"}:
        raise ValueError(f"train.early_stopping_mode must be 'min' or 'max', got: {early_mode}")
    progress_checkpoints = _normalize_progress_checkpoints(train_cfg.get("progress_checkpoints", None))

    prepared_dir = prepared_dir or (out_dir / "prepared")
    train_csv = prepared_dir / "train.csv"
    val_csv = prepared_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Prepared splits not found. Run prepare step first.")

    augment_cfg = {
        "positive_only": bool(train_cfg.get("augmentation_positive_only", True)),
        "apply_prob": float(train_cfg.get("augmentation_apply_prob", 1.0)),
        "noise_std": float(train_cfg.get("augmentation_noise_std", 0.0)),
        "scale_std": float(train_cfg.get("augmentation_scale_std", 0.0)),
        "feature_dropout_prob": float(train_cfg.get("augmentation_feature_dropout_prob", 0.0)),
        "time_dropout_prob": float(train_cfg.get("augmentation_time_dropout_prob", 0.0)),
        "value_dim": train_cfg.get("augmentation_value_dim", None),
        "value_clip": train_cfg.get("augmentation_value_clip", 8.0),
        "input_value_clip": train_cfg.get("input_value_clip", None),
        "input_value_dim": train_cfg.get("input_value_dim", None),
    }
    ds_train = PatientSequenceDataset(
        train_csv,
        max_patients=train_cfg.get("max_train_patients"),
        seed=seed + 11,
        augment=bool(train_cfg.get("augmentation_enabled", False)),
        augment_cfg=augment_cfg,
    )
    ds_val = PatientSequenceDataset(
        val_csv,
        max_patients=train_cfg.get("max_val_patients"),
        seed=seed + 17,
        augment_cfg=augment_cfg,
    )

    input_dim = ds_train[0]["x"].shape[1]
    model = build_model_from_config(input_dim=input_dim, model_cfg=model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sampler_mode = str(train_cfg.get("sampler_mode", "none")).lower()
    if sampler_mode not in {"none", "balanced_patient"}:
        raise ValueError(f"train.sampler_mode must be 'none' or 'balanced_patient', got: {sampler_mode}")

    train_sampler = None
    train_shuffle = True
    if sampler_mode == "balanced_patient":
        positive_fraction = float(train_cfg.get("sampler_positive_fraction", 0.5))
        weights = ds_train.balanced_sample_weights(positive_fraction=positive_fraction)
        raw_num_samples = train_cfg.get("sampler_num_samples", None)
        num_samples = (len(ds_train) if raw_num_samples is None else int(raw_num_samples))
        if num_samples <= 0:
            raise ValueError(f"train.sampler_num_samples must be > 0, got: {num_samples}")
        replacement = bool(train_cfg.get("sampler_replacement", True))
        gen = torch.Generator()
        gen.manual_seed(seed + 123)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
            generator=gen,
        )
        train_shuffle = False

    loader_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_patient_batch,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_patient_batch,
    )

    model_dir = model_dir or (out_dir / "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"
    history_path = model_dir / "train_history.json"
    latest_path = model_dir / "latest_checkpoint.pt"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val = float("inf")
    best_metric_value = (float("inf") if early_mode == "min" else float("-inf"))
    patience = int(train_cfg["early_stopping_patience"])
    patience_count = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    resume_mode = str(train_cfg.get("resume_mode", "none")).lower()
    if resume_mode not in {"none", "best", "latest"}:
        raise ValueError(
            f"train.resume_mode must be one of ['none','best','latest'], got: {resume_mode}"
        )
    resume_optimizer_state = bool(train_cfg.get("resume_optimizer_state", True))
    resume_path_cfg = train_cfg.get("resume_checkpoint_path", None)
    resume_path: Path | None = None
    if resume_path_cfg is not None:
        resume_path = Path(str(resume_path_cfg))
    elif resume_mode == "latest" and latest_path.exists():
        resume_path = latest_path
    elif resume_mode == "best" and best_path.exists():
        resume_path = best_path

    if history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded_history, list):
                history = loaded_history
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to load existing history at {history_path}: {e}")

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = ckpt_epoch + 1
        if "best_val_total" in checkpoint:
            best_val = float(checkpoint["best_val_total"])
        if "best_val_metric_value" in checkpoint:
            best_metric_value = float(checkpoint["best_val_metric_value"])
        if "patience_count" in checkpoint:
            patience_count = int(checkpoint["patience_count"])
        if resume_optimizer_state and ("optimizer_state_dict" in checkpoint):
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:  # pragma: no cover
                print(f"Warning: failed to restore optimizer state: {e}")
        print(f"Resuming from checkpoint={resume_path} at epoch={ckpt_epoch}")

    max_epochs = int(train_cfg["epochs"])
    if start_epoch > max_epochs:
        print(
            f"No training needed: start_epoch={start_epoch} exceeds configured epochs={max_epochs}."
        )
        return best_path

    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            device=device,
            lambda_propensity=float(train_cfg["lambda_propensity"]),
            lambda_balance=float(train_cfg["lambda_balance"]),
            lambda_smooth=float(train_cfg["lambda_smooth"]),
            lambda_sepsis=float(train_cfg.get("lambda_sepsis", 1.0)),
            sepsis_pos_weight=float(train_cfg.get("sepsis_pos_weight", 1.0)),
            l1_weight=l1_weight,
            l1_include_bias=l1_include_bias,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            metric_target=metric_target,
            metric_threshold=metric_threshold,
            patient_metric_aggregation=patient_metric_aggregation,
            patient_metric_threshold=patient_metric_threshold,
            sepsis_target_mode=sepsis_target_mode,
            sepsis_horizon_hours=sepsis_horizon_hours,
            compute_metrics=compute_train_metrics,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                loader=loader_val,
                optimizer=None,
                device=device,
                lambda_propensity=float(train_cfg["lambda_propensity"]),
                lambda_balance=float(train_cfg["lambda_balance"]),
                lambda_smooth=float(train_cfg["lambda_smooth"]),
                lambda_sepsis=float(train_cfg.get("lambda_sepsis", 1.0)),
                sepsis_pos_weight=float(train_cfg.get("sepsis_pos_weight", 1.0)),
                l1_weight=l1_weight,
                l1_include_bias=l1_include_bias,
                grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                metric_target=metric_target,
                metric_threshold=metric_threshold,
                patient_metric_aggregation=patient_metric_aggregation,
                patient_metric_threshold=patient_metric_threshold,
                sepsis_target_mode=sepsis_target_mode,
                sepsis_horizon_hours=sepsis_horizon_hours,
                compute_metrics=True,
            )

        entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(entry)
        # Persist history incrementally so long runs can be inspected/plotted even if interrupted.
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if early_metric not in val_metrics:
            raise ValueError(
                f"train.early_stopping_metric='{early_metric}' not found in val metrics keys: {list(val_metrics.keys())}"
            )
        current_metric = float(val_metrics[early_metric])
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_auroc={val_metrics['auroc']:.4f} "
            f"val_{early_metric}={current_metric:.4f}"
        )

        progress_rule = progress_checkpoints.get(epoch)
        if progress_rule is not None:
            fail_msg = _checkpoint_rule_failure(epoch=epoch, val_metrics=val_metrics, rule=progress_rule)
            if fail_msg is not None:
                print(f"Progress checkpoint failed at epoch={epoch}: {fail_msg}")
                print("Checkpoint guard triggered. Stopping early.")
                break

        is_improved = (
            (current_metric < best_metric_value)
            if early_mode == "min"
            else (current_metric > best_metric_value)
        )
        if is_improved:
            best_val = val_metrics["total"]
            best_metric_value = current_metric
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": normalize_model_config(input_dim=input_dim, model_cfg=model_cfg),
                    "train_config": train_cfg,
                    "best_val_total": best_val,
                    "best_val_metric_name": early_metric,
                    "best_val_metric_value": best_metric_value,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "patience_count": patience_count,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": normalize_model_config(input_dim=input_dim, model_cfg=model_cfg),
                "train_config": train_cfg,
                "best_val_total": best_val,
                "best_val_metric_name": early_metric,
                "best_val_metric_value": best_metric_value,
                "optimizer_state_dict": optimizer.state_dict(),
                "patience_count": patience_count,
                "epoch": epoch,
            },
            latest_path,
        )

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path
