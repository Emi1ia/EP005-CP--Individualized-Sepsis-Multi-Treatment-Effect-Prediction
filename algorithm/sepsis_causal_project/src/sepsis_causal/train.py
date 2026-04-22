from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import PatientSequenceDataset, collate_patient_batch
from .metrics import safe_classification_metrics
from .model import CausalTransformer, compute_losses
from .targets import build_temporal_target_torch


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _collect_masked(arr: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return arr[mask].detach().cpu().numpy()


def _run_epoch(
    model: CausalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_propensity: float,
    lambda_balance: float,
    lambda_smooth: float,
    lambda_sepsis: float,
    sepsis_pos_weight: float,
    grad_clip: float,
    metric_target: str,
    metric_threshold: float,
    sepsis_target_mode: str,
    sepsis_horizon_hours: int,
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
    }
    steps = 0

    y_true_list = []
    y_prob_list = []

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

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        for k in totals:
            totals[k] += float(loss_dict[k].detach().cpu())
        steps += 1

        if metric_target == "sepsis":
            y_true_list.append(_collect_masked(sepsis_target, batch["mask"]))
            y_prob_list.append(_collect_masked(loss_dict["sepsis_prob"], batch["mask"]))
        else:
            y_true_list.append(_collect_masked(batch["y"], batch["mask"]))
            y_prob_list.append(_collect_masked(loss_dict["factual_prob"], batch["mask"]))

    agg = {k: (v / max(steps, 1)) for k, v in totals.items()}
    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_prob = np.concatenate(y_prob_list) if y_prob_list else np.array([])
    cls_metrics = (
        safe_classification_metrics(y_true, y_prob, threshold=metric_threshold)
        if y_true.size
        else {"auroc": np.nan, "auprc": np.nan, "f1": np.nan}
    )
    agg.update(cls_metrics)
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
    sepsis_target_mode = str(train_cfg.get("sepsis_target_mode", "current")).lower()
    sepsis_horizon_hours = int(train_cfg.get("sepsis_horizon_hours", 0))
    early_metric = str(train_cfg.get("early_stopping_metric", "total"))
    early_mode = str(
        train_cfg.get(
            "early_stopping_mode",
            ("min" if early_metric == "total" else "max"),
        )
    ).lower()
    if early_mode not in {"min", "max"}:
        raise ValueError(f"train.early_stopping_mode must be 'min' or 'max', got: {early_mode}")

    prepared_dir = prepared_dir or (out_dir / "prepared")
    train_csv = prepared_dir / "train.csv"
    val_csv = prepared_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Prepared splits not found. Run prepare step first.")

    ds_train = PatientSequenceDataset(
        train_csv,
        max_patients=train_cfg.get("max_train_patients"),
        seed=seed + 11,
        augment=bool(train_cfg.get("augmentation_enabled", False)),
        augment_cfg={
            "positive_only": bool(train_cfg.get("augmentation_positive_only", True)),
            "apply_prob": float(train_cfg.get("augmentation_apply_prob", 1.0)),
            "noise_std": float(train_cfg.get("augmentation_noise_std", 0.0)),
            "scale_std": float(train_cfg.get("augmentation_scale_std", 0.0)),
            "feature_dropout_prob": float(train_cfg.get("augmentation_feature_dropout_prob", 0.0)),
            "time_dropout_prob": float(train_cfg.get("augmentation_time_dropout_prob", 0.0)),
            "value_dim": train_cfg.get("augmentation_value_dim", None),
            "value_clip": train_cfg.get("augmentation_value_clip", 8.0),
        },
    )
    ds_val = PatientSequenceDataset(
        val_csv,
        max_patients=train_cfg.get("max_val_patients"),
        seed=seed + 17,
    )

    input_dim = ds_train[0]["x"].shape[1]
    model = CausalTransformer(
        input_dim=input_dim,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    model_dir = model_dir or (out_dir / "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"
    history_path = model_dir / "train_history.json"

    best_val = float("inf")
    best_metric_value = (float("inf") if early_mode == "min" else float("-inf"))
    patience = int(train_cfg["early_stopping_patience"])
    patience_count = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
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
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            metric_target=metric_target,
            metric_threshold=metric_threshold,
            sepsis_target_mode=sepsis_target_mode,
            sepsis_horizon_hours=sepsis_horizon_hours,
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
                grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                metric_target=metric_target,
                metric_threshold=metric_threshold,
                sepsis_target_mode=sepsis_target_mode,
                sepsis_horizon_hours=sepsis_horizon_hours,
            )

        entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(entry)
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
                    "model_config": {
                        "input_dim": input_dim,
                        **model_cfg,
                    },
                    "train_config": train_cfg,
                    "best_val_total": best_val,
                    "best_val_metric_name": early_metric,
                    "best_val_metric_value": best_metric_value,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path
