from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a new prepared dataset with augmented positives to hit a target train ratio."
    )
    p.add_argument("--source-prepared-dir", type=str, required=True)
    p.add_argument("--target-prepared-dir", type=str, required=True)
    p.add_argument("--target-positive-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--apply-prob", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--scale-std", type=float, default=0.03)
    p.add_argument("--feature-dropout-prob", type=float, default=0.01)
    p.add_argument("--time-dropout-prob", type=float, default=0.0)
    p.add_argument("--value-dim", type=int, default=40)
    p.add_argument("--value-clip", type=float, default=8.0)
    return p.parse_args()


def _augment_x(
    x: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    scale_std: float,
    feature_dropout_prob: float,
    time_dropout_prob: float,
    value_dim: int,
    value_clip: float | None,
) -> np.ndarray:
    out = x.copy()
    vd = int(min(max(1, value_dim), out.shape[1]))
    xv = out[:, :vd]

    if scale_std > 0:
        scale = rng.normal(loc=1.0, scale=scale_std, size=(1, vd)).astype(np.float32)
        xv *= scale

    if noise_std > 0:
        xv += rng.normal(loc=0.0, scale=noise_std, size=xv.shape).astype(np.float32)

    if feature_dropout_prob > 0:
        mask = rng.random(size=xv.shape) < feature_dropout_prob
        xv[mask] = 0.0

    if time_dropout_prob > 0:
        t_mask = rng.random(size=(xv.shape[0],)) < time_dropout_prob
        xv[t_mask, :] = 0.0

    if value_clip is not None and value_clip > 0:
        np.clip(xv, -value_clip, value_clip, out=xv)

    out[:, :vd] = xv
    return out


def _required_positive_additions(n_pos: int, n_neg: int, target_pos_frac: float) -> int:
    n_total = n_pos + n_neg
    add = math.ceil((target_pos_frac * n_total - n_pos) / (1.0 - target_pos_frac))
    return max(0, int(add))


def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int64)


def _septic_mask(train_df: pd.DataFrame) -> np.ndarray:
    if "septic_patient" in train_df.columns:
        return _to_int_series(train_df["septic_patient"]).to_numpy() == 1

    flags: list[bool] = []
    for p in train_df["prepared_path"].tolist():
        payload = np.load(Path(str(p)))
        flags.append(bool(np.max(payload["sepsis_label"]) >= 1.0))
    return np.asarray(flags, dtype=bool)


def _validate_paths(source_dir: Path, target_dir: Path) -> None:
    required = ["train.csv", "val.csv", "test.csv"]
    for name in required:
        if not (source_dir / name).exists():
            raise FileNotFoundError(f"Missing required source file: {source_dir / name}")

    if target_dir.exists():
        existing = list(target_dir.glob("*"))
        if existing:
            raise FileExistsError(
                f"Target prepared dir is not empty: {target_dir}. "
                "Please choose a new directory."
            )


def main() -> None:
    args = _parse_args()

    source_dir = Path(args.source_prepared_dir).expanduser().resolve()
    target_dir = Path(args.target_prepared_dir).expanduser().resolve()
    target_pos_frac = float(args.target_positive_fraction)

    if not (0.0 < target_pos_frac < 1.0):
        raise ValueError("--target-positive-fraction must be in (0, 1)")

    _validate_paths(source_dir, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_patients_dir = target_dir / "patients"
    target_patients_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(source_dir / "train.csv")
    val_df = pd.read_csv(source_dir / "val.csv")
    test_df = pd.read_csv(source_dir / "test.csv")

    if "prepared_path" not in train_df.columns:
        raise ValueError("train.csv must contain a 'prepared_path' column")

    septic = _septic_mask(train_df)
    n_total = int(len(train_df))
    n_pos = int(np.sum(septic))
    n_neg = n_total - n_pos

    if n_pos == 0:
        raise ValueError("No positive rows in train split; cannot augment positives for balancing")

    add_pos = _required_positive_additions(n_pos=n_pos, n_neg=n_neg, target_pos_frac=target_pos_frac)

    rng = np.random.default_rng(int(args.seed))
    pos_df = train_df[septic].reset_index(drop=True)
    new_rows: list[dict[str, Any]] = []

    if "patient_id" in train_df.columns:
        next_patient_id = int(_to_int_series(train_df["patient_id"]).max()) + 1
    else:
        next_patient_id = 1

    print(
        "source_train_total=", n_total,
        " source_pos=", n_pos,
        " source_neg=", n_neg,
        " add_pos=", add_pos,
    )

    for i in range(add_pos):
        src_row = pos_df.iloc[int(rng.integers(0, len(pos_df)))]
        src_payload_path = Path(str(src_row["prepared_path"]))
        payload = np.load(src_payload_path)

        x = payload["x"].astype(np.float32)
        if rng.random() <= float(np.clip(args.apply_prob, 0.0, 1.0)):
            x = _augment_x(
                x=x,
                rng=rng,
                noise_std=float(max(0.0, args.noise_std)),
                scale_std=float(max(0.0, args.scale_std)),
                feature_dropout_prob=float(np.clip(args.feature_dropout_prob, 0.0, 1.0)),
                time_dropout_prob=float(np.clip(args.time_dropout_prob, 0.0, 1.0)),
                value_dim=int(max(1, args.value_dim)),
                value_clip=None if args.value_clip is None else float(max(0.0, args.value_clip)),
            )

        source_pid = int(src_row["patient_id"]) if "patient_id" in src_row.index else 0
        out_name = f"aug_p{source_pid:07d}_{i + 1:05d}.npz"
        out_path = (target_patients_dir / out_name).resolve()
        np.savez_compressed(
            out_path,
            x=x.astype(np.float32),
            actions=payload["actions"].astype(np.int64),
            y=payload["y"].astype(np.float32),
            y_all=payload["y_all"].astype(np.float32),
            sepsis_label=payload["sepsis_label"].astype(np.int64),
        )

        row_dict = src_row.to_dict()
        if "patient_id" in train_df.columns:
            row_dict["patient_id"] = int(next_patient_id)
            next_patient_id += 1
        if "split" in train_df.columns:
            row_dict["split"] = "train"
        if "prepared_path" in train_df.columns:
            row_dict["prepared_path"] = str(out_path)
        if "length" in train_df.columns:
            row_dict["length"] = int(x.shape[0])
        if "septic_patient" in train_df.columns:
            row_dict["septic_patient"] = 1
        new_rows.append(row_dict)

        if (i + 1) % 2000 == 0:
            print(f"augmented {i + 1}/{add_pos}")

    if new_rows:
        add_df = pd.DataFrame(new_rows, columns=train_df.columns)
        train_out = pd.concat([train_df, add_df], axis=0, ignore_index=True)
    else:
        train_out = train_df.copy()

    if "patient_id" in train_out.columns:
        train_out["patient_id"] = _to_int_series(train_out["patient_id"])
        train_out = train_out.sort_values("patient_id").reset_index(drop=True)

    train_out.to_csv(target_dir / "train.csv", index=False)
    val_df.to_csv(target_dir / "val.csv", index=False)
    test_df.to_csv(target_dir / "test.csv", index=False)

    manifest = pd.concat([train_out, val_df, test_df], axis=0, ignore_index=True)
    manifest.to_csv(target_dir / "manifest.csv", index=False)

    src_stats = source_dir / "normalization_stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, target_dir / "normalization_stats.json")

    out_septic = _septic_mask(train_out)
    out_total = int(len(train_out))
    out_pos = int(np.sum(out_septic))
    out_neg = int(out_total - out_pos)

    report = {
        "source_prepared_dir": str(source_dir),
        "target_prepared_dir": str(target_dir),
        "target_positive_fraction": target_pos_frac,
        "source_train_total": n_total,
        "source_train_pos": n_pos,
        "source_train_neg": n_neg,
        "added_augmented_positive_rows": int(add_pos),
        "output_train_total": out_total,
        "output_train_pos": out_pos,
        "output_train_neg": out_neg,
        "output_train_pos_fraction": (float(out_pos) / float(out_total) if out_total else 0.0),
        "augmentation_config": {
            "seed": int(args.seed),
            "apply_prob": float(args.apply_prob),
            "noise_std": float(args.noise_std),
            "scale_std": float(args.scale_std),
            "feature_dropout_prob": float(args.feature_dropout_prob),
            "time_dropout_prob": float(args.time_dropout_prob),
            "value_dim": int(args.value_dim),
            "value_clip": float(args.value_clip),
        },
    }
    (target_dir / "augmentation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
