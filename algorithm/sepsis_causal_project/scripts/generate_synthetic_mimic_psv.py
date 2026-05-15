from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_causal.constants import FEATURE_COLS, LABEL_COL


STATIC_FEATURES = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}

# (low, high) clipping ranges for physiologic plausibility.
RANGES: dict[str, tuple[float, float]] = {
    "HR": (35.0, 210.0),
    "O2Sat": (55.0, 100.0),
    "Temp": (33.0, 41.5),
    "SBP": (60.0, 220.0),
    "MAP": (40.0, 160.0),
    "DBP": (30.0, 130.0),
    "Resp": (6.0, 55.0),
    "EtCO2": (10.0, 70.0),
    "BaseExcess": (-25.0, 20.0),
    "HCO3": (8.0, 45.0),
    "FiO2": (0.21, 1.0),
    "pH": (6.9, 7.7),
    "PaCO2": (15.0, 95.0),
    "SaO2": (50.0, 100.0),
    "AST": (5.0, 1000.0),
    "BUN": (2.0, 160.0),
    "Alkalinephos": (20.0, 1200.0),
    "Calcium": (4.0, 14.0),
    "Chloride": (70.0, 130.0),
    "Creatinine": (0.2, 15.0),
    "Bilirubin_direct": (0.0, 20.0),
    "Glucose": (35.0, 600.0),
    "Lactate": (0.2, 20.0),
    "Magnesium": (0.8, 6.0),
    "Phosphate": (0.5, 12.0),
    "Potassium": (2.0, 8.0),
    "Bilirubin_total": (0.0, 40.0),
    "TroponinI": (0.0, 80.0),
    "Hct": (10.0, 60.0),
    "Hgb": (4.0, 20.0),
    "PTT": (18.0, 180.0),
    "WBC": (0.5, 80.0),
    "Fibrinogen": (50.0, 1000.0),
    "Platelets": (5.0, 900.0),
}

# Approximate observation probabilities per hour.
OBS_PROB: dict[str, float] = {
    "HR": 0.92,
    "O2Sat": 0.90,
    "Temp": 0.45,
    "SBP": 0.88,
    "MAP": 0.88,
    "DBP": 0.88,
    "Resp": 0.86,
    "EtCO2": 0.12,
    "BaseExcess": 0.08,
    "HCO3": 0.10,
    "FiO2": 0.20,
    "pH": 0.11,
    "PaCO2": 0.10,
    "SaO2": 0.07,
    "AST": 0.04,
    "BUN": 0.20,
    "Alkalinephos": 0.03,
    "Calcium": 0.14,
    "Chloride": 0.16,
    "Creatinine": 0.20,
    "Bilirubin_direct": 0.02,
    "Glucose": 0.26,
    "Lactate": 0.11,
    "Magnesium": 0.13,
    "Phosphate": 0.10,
    "Potassium": 0.20,
    "Bilirubin_total": 0.06,
    "TroponinI": 0.02,
    "Hct": 0.23,
    "Hgb": 0.23,
    "PTT": 0.10,
    "WBC": 0.20,
    "Fibrinogen": 0.04,
    "Platelets": 0.21,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic MIMIC-like patient files (p*.psv) in the PhysioNet schema "
            f"({len(FEATURE_COLS)} features + {LABEL_COL})."
        )
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-patients", type=int, default=2000)
    p.add_argument("--positive-rate", type=float, default=0.18)
    p.add_argument("--min-hours", type=int, default=12)
    p.add_argument("--max-hours", type=int, default=96)
    p.add_argument("--patient-id-offset", type=int, default=9_100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _clip(name: str, x: np.ndarray) -> np.ndarray:
    lo, hi = RANGES[name]
    return np.clip(x, lo, hi)


def _severity_curve(t: int, onset: int | None) -> np.ndarray:
    if onset is None:
        return np.zeros((t,), dtype=np.float32)
    h = np.arange(1, t + 1, dtype=np.float32)
    z = (h - float(onset)) / 5.0
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _apply_missingness(
    rng: np.random.Generator,
    name: str,
    values: np.ndarray,
) -> np.ndarray:
    p = float(OBS_PROB.get(name, 0.15))
    mask = rng.random(size=values.shape[0]) < p
    out = values.astype(np.float32, copy=True)
    out[~mask] = np.nan
    return out


def _sample_time_series(
    rng: np.random.Generator,
    t: int,
    septic: bool,
    severity: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline_jitter = rng.normal(loc=0.0, scale=1.0, size=(t,)).astype(np.float32)
    circadian = (0.8 * np.sin(np.arange(1, t + 1, dtype=np.float32) / 6.0)).astype(np.float32)

    data: dict[str, np.ndarray] = {}
    s = severity if septic else np.zeros_like(severity)

    data["HR"] = _clip("HR", 85.0 + 18.0 * s + 4.0 * circadian + 7.0 * baseline_jitter)
    data["O2Sat"] = _clip("O2Sat", 97.0 - 6.0 * s + 1.2 * baseline_jitter)
    data["Temp"] = _clip("Temp", 36.8 + 1.1 * s + 0.18 * baseline_jitter)
    data["SBP"] = _clip("SBP", 118.0 - 18.0 * s + 8.0 * baseline_jitter)
    data["MAP"] = _clip("MAP", 80.0 - 13.0 * s + 6.0 * baseline_jitter)
    data["DBP"] = _clip("DBP", 66.0 - 10.0 * s + 5.0 * baseline_jitter)
    data["Resp"] = _clip("Resp", 18.0 + 7.0 * s + 2.5 * baseline_jitter)
    data["EtCO2"] = _clip("EtCO2", 35.0 - 5.0 * s + 3.0 * baseline_jitter)
    data["BaseExcess"] = _clip("BaseExcess", -1.0 - 4.5 * s + 1.3 * baseline_jitter)
    data["HCO3"] = _clip("HCO3", 24.0 - 3.0 * s + 1.6 * baseline_jitter)
    data["FiO2"] = _clip("FiO2", 0.25 + 0.20 * s + 0.03 * baseline_jitter)
    data["pH"] = _clip("pH", 7.39 - 0.07 * s + 0.015 * baseline_jitter)
    data["PaCO2"] = _clip("PaCO2", 40.0 + 7.0 * s + 2.0 * baseline_jitter)
    data["SaO2"] = _clip("SaO2", 96.0 - 5.0 * s + 1.3 * baseline_jitter)
    data["AST"] = _clip("AST", 34.0 + 75.0 * s + 20.0 * np.abs(baseline_jitter))
    data["BUN"] = _clip("BUN", 20.0 + 10.0 * s + 4.0 * np.abs(baseline_jitter))
    data["Alkalinephos"] = _clip("Alkalinephos", 105.0 + 35.0 * s + 14.0 * np.abs(baseline_jitter))
    data["Calcium"] = _clip("Calcium", 8.9 - 0.7 * s + 0.25 * baseline_jitter)
    data["Chloride"] = _clip("Chloride", 103.0 + 2.0 * s + 2.5 * baseline_jitter)
    data["Creatinine"] = _clip("Creatinine", 1.0 + 0.7 * s + 0.25 * np.abs(baseline_jitter))
    data["Bilirubin_direct"] = _clip(
        "Bilirubin_direct", 0.2 + 0.6 * s + 0.2 * np.abs(baseline_jitter)
    )
    data["Glucose"] = _clip("Glucose", 120.0 + 55.0 * s + 20.0 * baseline_jitter)
    data["Lactate"] = _clip("Lactate", 1.4 + 3.0 * s + 0.5 * np.abs(baseline_jitter))
    data["Magnesium"] = _clip("Magnesium", 2.0 + 0.45 * s + 0.2 * baseline_jitter)
    data["Phosphate"] = _clip("Phosphate", 3.6 + 0.6 * s + 0.35 * baseline_jitter)
    data["Potassium"] = _clip("Potassium", 4.0 + 0.45 * s + 0.25 * baseline_jitter)
    data["Bilirubin_total"] = _clip("Bilirubin_total", 0.8 + 1.4 * s + 0.35 * np.abs(baseline_jitter))
    data["TroponinI"] = _clip("TroponinI", 0.04 + 0.9 * s + 0.2 * np.abs(baseline_jitter))
    data["Hct"] = _clip("Hct", 35.0 - 4.0 * s + 1.8 * baseline_jitter)
    data["Hgb"] = _clip("Hgb", 11.8 - 1.4 * s + 0.8 * baseline_jitter)
    data["PTT"] = _clip("PTT", 31.0 + 12.0 * s + 5.0 * np.abs(baseline_jitter))
    data["WBC"] = _clip("WBC", 9.5 + 6.5 * s + 2.5 * np.abs(baseline_jitter))
    data["Fibrinogen"] = _clip("Fibrinogen", 380.0 - 110.0 * s + 35.0 * baseline_jitter)
    data["Platelets"] = _clip("Platelets", 230.0 - 95.0 * s + 35.0 * baseline_jitter)

    for k in list(data.keys()):
        data[k] = _apply_missingness(rng=rng, name=k, values=data[k])
    return data


def _build_patient_frame(
    rng: np.random.Generator,
    t: int,
    septic: bool,
) -> tuple[pd.DataFrame, int | None]:
    frame = pd.DataFrame(index=np.arange(t), columns=FEATURE_COLS, dtype=np.float32)

    age = float(rng.integers(18, 91))
    gender = float(rng.integers(0, 2))
    unit1 = 1.0 if rng.random() < 0.55 else 0.0
    unit2 = 1.0 - unit1
    hosp_adm_time = -float(rng.integers(1, 24 * 10 + 1))

    frame["Age"] = age
    frame["Gender"] = gender
    frame["Unit1"] = unit1
    frame["Unit2"] = unit2
    frame["HospAdmTime"] = hosp_adm_time
    frame["ICULOS"] = np.arange(1, t + 1, dtype=np.int64)

    onset: int | None = None
    if septic:
        start_min = 4
        start_max = max(5, t - 4)
        onset = int(rng.integers(start_min, start_max + 1))

    severity = _severity_curve(t=t, onset=onset)
    dyn = _sample_time_series(rng=rng, t=t, septic=septic, severity=severity)
    for col, vals in dyn.items():
        frame[col] = vals

    label = np.zeros((t,), dtype=np.int64)
    if onset is not None:
        label[onset - 1 :] = 1
    frame[LABEL_COL] = label
    return frame, onset


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("p*.psv"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output dir already has {len(existing)} patient files: {out_dir}. "
            "Use --overwrite to replace them."
        )
    if args.overwrite:
        for fp in out_dir.glob("p*.psv"):
            fp.unlink()

    if args.n_patients <= 0:
        raise ValueError("--n-patients must be > 0")
    if not (0.0 < args.positive_rate < 1.0):
        raise ValueError("--positive-rate must be in (0, 1)")
    if args.min_hours < 1 or args.max_hours < args.min_hours:
        raise ValueError("--min-hours must be >=1 and <= --max-hours")

    rng = np.random.default_rng(int(args.seed))
    manifest_rows: list[dict[str, object]] = []

    septic_count = 0
    total_hours = 0
    missing_num = 0
    missing_den = 0

    for i in range(int(args.n_patients)):
        pid = int(args.patient_id_offset) + i + 1
        t = int(rng.integers(int(args.min_hours), int(args.max_hours) + 1))
        septic = bool(rng.random() < float(args.positive_rate))

        frame, onset = _build_patient_frame(rng=rng, t=t, septic=septic)
        out_path = out_dir / f"p{pid:07d}.psv"
        frame.to_csv(out_path, sep="|", index=False)

        septic_count += int(septic)
        total_hours += t

        vals = frame[FEATURE_COLS].to_numpy(dtype=np.float64)
        missing_num += int(np.isnan(vals).sum())
        missing_den += int(vals.size)

        manifest_rows.append(
            {
                "patient_id": pid,
                "path": str(out_path),
                "length_hours": t,
                "septic_patient": int(septic),
                "onset_hour": (int(onset) if onset is not None else None),
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"generated {i + 1}/{args.n_patients}")

    manifest_path = out_dir / "synthetic_mimic_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "dataset_type": "synthetic_mimic_like",
        "note": (
            "Synthetic data only. Not real MIMIC-IV records. "
            "Use for method development/testing, not clinical claims."
        ),
        "output_dir": str(out_dir),
        "n_patients": int(args.n_patients),
        "n_septic_patients": int(septic_count),
        "septic_fraction": float(septic_count / max(1, int(args.n_patients))),
        "avg_length_hours": float(total_hours / max(1, int(args.n_patients))),
        "feature_missing_fraction": float(missing_num / max(1, missing_den)),
        "min_hours": int(args.min_hours),
        "max_hours": int(args.max_hours),
        "patient_id_offset": int(args.patient_id_offset),
        "seed": int(args.seed),
        "manifest_path": str(manifest_path),
    }
    (out_dir / "synthetic_mimic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

