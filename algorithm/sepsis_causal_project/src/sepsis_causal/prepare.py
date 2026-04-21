from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FEATURE_COLS, LABEL_COL
from .io import list_patient_files, load_patient_frame, split_patient_files
from .simulation import derive_severity_terms, factual_from_actions, generate_potential_outcomes, sample_actions
from .utils import parse_patient_id


def _fit_normalization_stats(train_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_n = len(FEATURE_COLS)
    obs_count = np.zeros(feat_n, dtype=np.float64)
    obs_sum = np.zeros(feat_n, dtype=np.float64)
    obs_sq_sum = np.zeros(feat_n, dtype=np.float64)

    for i, fp in enumerate(train_files, 1):
        df = load_patient_frame(fp)
        x = df[FEATURE_COLS].to_numpy(dtype=np.float64, copy=False)
        m = ~np.isnan(x)
        obs_count += m.sum(axis=0)
        obs_sum += np.where(m, x, 0.0).sum(axis=0)
        obs_sq_sum += np.where(m, x * x, 0.0).sum(axis=0)
        if i % 2000 == 0:
            print(f"normalization scan: {i}/{len(train_files)}")

    valid = obs_count > 0
    mean = np.zeros(feat_n, dtype=np.float64)
    std = np.ones(feat_n, dtype=np.float64)
    mean[valid] = obs_sum[valid] / obs_count[valid]
    var = np.zeros(feat_n, dtype=np.float64)
    var[valid] = (obs_sq_sum[valid] / obs_count[valid]) - mean[valid] ** 2
    var = np.maximum(var, 1e-6)
    std[valid] = np.sqrt(var[valid])
    return mean.astype(np.float32), std.astype(np.float32), obs_count.astype(np.int64)


def _forward_fill_then_mean_impute(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = x.copy()
    t, f = out.shape
    for j in range(f):
        last = np.nan
        for i in range(t):
            if np.isnan(out[i, j]):
                if not np.isnan(last):
                    out[i, j] = last
            else:
                last = out[i, j]
        nan_mask = np.isnan(out[:, j])
        if nan_mask.any():
            out[nan_mask, j] = mean[j]
    return out


def _compute_delta(mask: np.ndarray, delta_cap_hours: float) -> np.ndarray:
    t, f = mask.shape
    delta = np.zeros((t, f), dtype=np.float32)
    for j in range(f):
        since = 0.0
        for i in range(t):
            if mask[i, j]:
                since = 0.0
            else:
                since += 1.0
            delta[i, j] = since
    delta = np.minimum(delta, delta_cap_hours) / float(delta_cap_hours)
    return delta


def _build_patient_arrays(
    df: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    delta_cap_hours: float,
    synthetic_noise_std: float,
    seed: int,
    patient_id: int,
) -> dict[str, np.ndarray]:
    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    sepsis = df[LABEL_COL].to_numpy(dtype=np.int64, copy=False)

    obs_mask = ~np.isnan(raw)
    filled = _forward_fill_then_mean_impute(raw, mean)
    x_norm = (filled - mean) / std
    delta = _compute_delta(obs_mask, delta_cap_hours)

    x_input = np.concatenate(
        [x_norm.astype(np.float32), obs_mask.astype(np.float32), delta.astype(np.float32)], axis=1
    )

    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    terms = derive_severity_terms(filled.astype(np.float64), col_idx)
    rng = np.random.default_rng(seed + patient_id * 1009)
    actions = sample_actions(terms, rng)
    y_all = generate_potential_outcomes(terms, synthetic_noise_std, rng)
    y = factual_from_actions(y_all, actions, rng)

    return {
        "x": x_input.astype(np.float32),
        "actions": actions.astype(np.int64),
        "y": y.astype(np.float32),
        "y_all": y_all.astype(np.float32),
        "sepsis_label": sepsis.astype(np.int64),
    }


def run_prepare(
    config: dict[str, Any],
    data_root: Path,
    out_dir: Path,
) -> Path:
    train_dirs = config["data"]["train_dirs"]
    max_patients = config["data"]["max_patients"]
    seed = int(config["seed"])
    split_ratio = config["data"]["split_ratio"]
    delta_cap_hours = float(config["data"]["delta_cap_hours"])
    synthetic_noise_std = float(config["data"]["synthetic_noise_std"])

    all_files = list_patient_files(data_root, train_dirs)
    if max_patients is not None:
        all_files = all_files[: int(max_patients)]
    if not all_files:
        raise ValueError("No patient files found for preparation")

    split_files = split_patient_files(
        all_files,
        seed=seed,
        train_ratio=float(split_ratio["train"]),
        val_ratio=float(split_ratio["val"]),
        test_ratio=float(split_ratio["test"]),
    )

    print("Fitting normalization stats on train split...")
    mean, std, obs_count = _fit_normalization_stats(split_files["train"])

    prepared_dir = out_dir / "prepared"
    patient_dir = prepared_dir / "patients"
    patient_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = {
        "feature_cols": FEATURE_COLS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "observed_count": obs_count.tolist(),
    }
    (prepared_dir / "normalization_stats.json").write_text(
        json.dumps(stats_payload, indent=2), encoding="utf-8"
    )

    for split_name, files in split_files.items():
        records: list[dict[str, Any]] = []
        print(f"Preparing split={split_name} with {len(files)} patients")
        for i, fp in enumerate(files, 1):
            df = load_patient_frame(fp)
            patient_id = parse_patient_id(fp)

            arrays = _build_patient_arrays(
                df=df,
                mean=mean,
                std=std,
                delta_cap_hours=delta_cap_hours,
                synthetic_noise_std=synthetic_noise_std,
                seed=seed,
                patient_id=patient_id,
            )

            out_npz = patient_dir / f"{fp.stem}.npz"
            np.savez_compressed(
                out_npz,
                x=arrays["x"],
                actions=arrays["actions"],
                y=arrays["y"],
                y_all=arrays["y_all"],
                sepsis_label=arrays["sepsis_label"],
            )

            records.append(
                {
                    "patient_id": patient_id,
                    "split": split_name,
                    "raw_path": str(fp),
                    "prepared_path": str(out_npz),
                    "length": int(arrays["x"].shape[0]),
                    "septic_patient": int(arrays["sepsis_label"].max()),
                }
            )
            if i % 1000 == 0:
                print(f"  {split_name}: {i}/{len(files)}")

        pd.DataFrame(records).sort_values("patient_id").to_csv(
            prepared_dir / f"{split_name}.csv", index=False
        )

    manifest = pd.concat(
        [
            pd.read_csv(prepared_dir / "train.csv"),
            pd.read_csv(prepared_dir / "val.csv"),
            pd.read_csv(prepared_dir / "test.csv"),
        ],
        axis=0,
        ignore_index=True,
    )
    manifest.to_csv(prepared_dir / "manifest.csv", index=False)
    return prepared_dir

