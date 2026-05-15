from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_plot_history(
    config: dict[str, Any],
    out_dir: Path,
    history_path: Path | None = None,
    plot_path: Path | None = None,
) -> Path:
    history_path = history_path or (out_dir / "model" / "train_history.json")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing train history file: {history_path}")

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        raise ValueError(f"Unexpected train history format in {history_path}: expected non-empty list")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    plot_path = plot_path or (out_dir / "eval" / "epoch_error_curves.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(entry.get("epoch", idx + 1)) for idx, entry in enumerate(history)]
    metric_names = ("mae", "mse", "rmse")

    train_series = {
        metric: [_to_float(entry.get("train", {}).get(metric, float("nan"))) for entry in history]
        for metric in metric_names
    }
    val_series = {
        metric: [_to_float(entry.get("val", {}).get(metric, float("nan"))) for entry in history]
        for metric in metric_names
    }

    has_any_metric = any(
        np.isfinite(np.asarray(train_series[metric], dtype=np.float64)).any()
        or np.isfinite(np.asarray(val_series[metric], dtype=np.float64)).any()
        for metric in metric_names
    )
    if not has_any_metric:
        raise ValueError(
            "No per-epoch MAE/MSE/RMSE found in train_history.json. "
            "Retrain with the current code to record these metrics."
        )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, metric in zip(axes, metric_names, strict=True):
        t = np.asarray(train_series[metric], dtype=np.float64)
        v = np.asarray(val_series[metric], dtype=np.float64)

        if np.isfinite(t).any():
            ax.plot(epochs, t, label="train", linewidth=1.8)
        if np.isfinite(v).any():
            ax.plot(epochs, v, label="val", linewidth=1.8)

        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle="--", alpha=0.35)
        if ax.lines:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Per-Epoch Error Metrics")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path
