from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

from .evaluate import run_evaluate_with_paths
from .model import SUPPORTED_MODEL_TYPES
from .prepare import run_prepare
from .train import run_train_with_paths
from .utils import save_json


def _resolve_optional_path(path_value: str | None, default_path: Path) -> Path:
    if path_value is None:
        return default_path.resolve()
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def _metric(payload: dict[str, Any], *keys: str) -> float | None:
    cur: Any = payload
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _load_json(path: Path) -> dict[str, Any]:
    last_err: Exception | None = None
    for _ in range(6):
        try:
            txt = path.read_text(encoding="utf-8")
            if not txt.strip():
                raise json.JSONDecodeError("empty file", txt, 0)
            return json.loads(txt)
        except (json.JSONDecodeError, OSError) as e:
            last_err = e
            time.sleep(0.25)
    raise RuntimeError(f"Could not read valid JSON from {path}: {last_err}")


def _summary_row(model_type: str, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "run_dir": str(run_dir),
        "checkpoint": metrics.get("checkpoint"),
        "n_test_sequences": metrics.get("n_test_sequences"),
        "n_test_timesteps": metrics.get("n_test_timesteps"),
        "sepsis_auroc": _metric(metrics, "sepsis_metrics", "auroc"),
        "sepsis_auprc": _metric(metrics, "sepsis_metrics", "auprc"),
        "sepsis_f1": _metric(metrics, "sepsis_metrics", "f1"),
        "patient_sepsis_auroc": _metric(metrics, "patient_sepsis_metrics", "auroc"),
        "patient_sepsis_auprc": _metric(metrics, "patient_sepsis_metrics", "auprc"),
        "patient_sepsis_f1": _metric(metrics, "patient_sepsis_metrics", "f1"),
        "sepsis_mae": _metric(metrics, "sepsis_error_metrics", "mae"),
        "sepsis_rmse": _metric(metrics, "sepsis_error_metrics", "rmse"),
        "patient_sepsis_mae": _metric(metrics, "patient_sepsis_error_metrics", "mae"),
        "patient_sepsis_rmse": _metric(metrics, "patient_sepsis_error_metrics", "rmse"),
        "pehe": _metric(metrics, "treatment_effect_metrics", "pehe"),
        "ate_error": _metric(metrics, "treatment_effect_metrics", "ate_error"),
        "policy_regret": _metric(metrics, "treatment_effect_metrics", "policy_regret"),
    }


def _rank_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    # Higher AUPRC/F1 is better; lower PEHE is better.
    auprc = entry.get("patient_sepsis_auprc")
    f1 = entry.get("patient_sepsis_f1")
    pehe = entry.get("pehe")
    return (
        -(auprc if isinstance(auprc, (int, float)) else -1.0),
        -(f1 if isinstance(f1, (int, float)) else -1.0),
        (pehe if isinstance(pehe, (int, float)) else 1e9),
    )


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "model_type",
        "run_dir",
        "checkpoint",
        "n_test_sequences",
        "n_test_timesteps",
        "sepsis_auroc",
        "sepsis_auprc",
        "sepsis_f1",
        "patient_sepsis_auroc",
        "patient_sepsis_auprc",
        "patient_sepsis_f1",
        "sepsis_mae",
        "sepsis_rmse",
        "patient_sepsis_mae",
        "patient_sepsis_rmse",
        "pehe",
        "ate_error",
        "policy_regret",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            out = {k: row.get(k) for k in fieldnames if k != "rank"}
            out["rank"] = i
            writer.writerow(out)


def run_model_benchmark(config: dict[str, Any], data_root: Path, out_dir: Path) -> Path:
    benchmark_cfg = config.get("benchmark", {})
    model_types_raw = benchmark_cfg.get("model_types", list(SUPPORTED_MODEL_TYPES))
    if not isinstance(model_types_raw, list) or not model_types_raw:
        raise ValueError("benchmark.model_types must be a non-empty list")

    model_types = [str(m).lower() for m in model_types_raw]
    invalid = [m for m in model_types if m not in SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError(
            f"Unsupported benchmark model types: {invalid}. Supported: {list(SUPPORTED_MODEL_TYPES)}"
        )

    benchmark_root_default = out_dir.parent / f"{out_dir.name}_model_benchmarks"
    benchmark_root = _resolve_optional_path(benchmark_cfg.get("out_dir"), benchmark_root_default)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    prepared_dir_override = benchmark_cfg.get("prepared_dir")
    prepared_dir = _resolve_optional_path(prepared_dir_override, out_dir / "prepared")

    auto_prepare = bool(benchmark_cfg.get("auto_prepare", True))
    train_csv = prepared_dir / "train.csv"
    val_csv = prepared_dir / "val.csv"
    test_csv = prepared_dir / "test.csv"
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        if not auto_prepare:
            raise FileNotFoundError(
                f"Prepared split files not found under {prepared_dir}. "
                "Set benchmark.auto_prepare=true or run prepare first."
            )
        prep_dir = run_prepare(config=config, data_root=data_root, out_dir=out_dir)
        prepared_dir = prep_dir
        train_csv = prepared_dir / "train.csv"
        val_csv = prepared_dir / "val.csv"
        test_csv = prepared_dir / "test.csv"

    model_overrides = benchmark_cfg.get("model_overrides", {})
    train_overrides = benchmark_cfg.get("train_overrides", {})
    skip_existing = bool(benchmark_cfg.get("skip_existing", True))

    rows: list[dict[str, Any]] = []
    run_details: list[dict[str, Any]] = []

    for model_type in model_types:
        run_cfg = copy.deepcopy(config)
        run_cfg.setdefault("model", {})
        run_cfg["model"]["model_type"] = model_type

        model_patch = model_overrides.get(model_type, {}) if isinstance(model_overrides, dict) else {}
        if isinstance(model_patch, dict):
            run_cfg["model"].update(model_patch)

        run_cfg.setdefault("train", {})
        if isinstance(train_overrides, dict):
            run_cfg["train"].update(train_overrides)

        run_dir = benchmark_root / f"run_{model_type}"
        metrics_path = run_dir / "eval" / "metrics.json"

        if not (skip_existing and metrics_path.exists()):
            best_model = run_train_with_paths(
                config=run_cfg,
                out_dir=run_dir,
                prepared_dir=prepared_dir,
                model_dir=run_dir / "model",
            )
            metrics_path = run_evaluate_with_paths(
                config=run_cfg,
                out_dir=run_dir,
                prepared_dir=prepared_dir,
                model_path=best_model,
            )

        metrics = _load_json(metrics_path)
        row = _summary_row(model_type=model_type, run_dir=run_dir, metrics=metrics)
        rows.append(row)
        run_details.append(
            {
                "model_type": model_type,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
        )

    rows_sorted = sorted(rows, key=_rank_key)

    summary = {
        "benchmark_root": str(benchmark_root),
        "prepared_dir": str(prepared_dir),
        "model_types": model_types,
        "rank_sort": [
            "patient_sepsis_auprc desc",
            "patient_sepsis_f1 desc",
            "pehe asc",
        ],
        "runs": run_details,
        "results_ranked": rows_sorted,
    }

    summary_json = benchmark_root / "comparison_model_families.json"
    summary_csv = benchmark_root / "comparison_model_families.csv"
    save_json(summary, summary_json)
    _write_csv(rows_sorted, summary_csv)

    return summary_json
