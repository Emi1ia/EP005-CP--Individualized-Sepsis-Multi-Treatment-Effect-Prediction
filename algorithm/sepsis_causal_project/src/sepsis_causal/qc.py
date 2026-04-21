from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import FEATURE_COLS, LABEL_COL
from .io import list_patient_files, load_patient_frame
from .utils import save_json


PLAUSIBILITY_RANGES = {
    "HR": (20.0, 260.0),
    "O2Sat": (40.0, 100.0),
    "Temp": (30.0, 43.0),
    "SBP": (40.0, 300.0),
    "MAP": (25.0, 220.0),
    "DBP": (20.0, 180.0),
    "Resp": (4.0, 80.0),
    "pH": (6.8, 7.8),
    "FiO2": (0.21, 1.0),
    "Lactate": (0.1, 30.0),
    "Creatinine": (0.1, 20.0),
    "WBC": (0.1, 150.0),
    "Platelets": (1.0, 1500.0),
}

EXPECTED_TREATMENT_COLS = [
    "Antibiotic",
    "Antibiotics",
    "IVFluid",
    "FluidBolus",
    "Vasopressor",
    "Norepinephrine",
]


def _safe_stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {"count": 0, "p25": None, "median": None, "p75": None, "mean": None}
    return {
        "count": int(arr.size),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(arr.mean()),
    }


def build_quality_report(
    data_root: Path,
    train_dirs: list[str],
    max_patients: int | None = None,
) -> dict[str, Any]:
    files = list_patient_files(data_root, train_dirs)
    if max_patients is not None:
        files = files[:max_patients]

    if not files:
        raise ValueError("No patient files found")

    ncol = len(FEATURE_COLS) + 1
    col_non_nan = np.zeros(ncol, dtype=np.int64)
    col_total = np.zeros(ncol, dtype=np.int64)
    col_min = np.full(ncol, np.inf)
    col_max = np.full(ncol, -np.inf)

    lengths = []
    patient_septic = 0
    onset_hours = []
    row_total = 0
    row_positive = 0
    reversion_patients = 0
    non_monotonic_iculos = 0
    observed_col_name_set = set()

    plaus_counts = {k: {"outside": 0, "observed": 0} for k in PLAUSIBILITY_RANGES}

    for fp in files:
        df = load_patient_frame(fp)
        observed_col_name_set.update(df.columns.tolist())

        arr = df[FEATURE_COLS + [LABEL_COL]].to_numpy(dtype=np.float64, copy=False)
        mask = ~np.isnan(arr)
        n = len(df)
        lengths.append(n)
        row_total += n

        col_non_nan += mask.sum(axis=0)
        col_total += n

        for j in range(arr.shape[1]):
            mj = mask[:, j]
            if mj.any():
                vj = arr[mj, j]
                vmin = vj.min()
                vmax = vj.max()
                if vmin < col_min[j]:
                    col_min[j] = vmin
                if vmax > col_max[j]:
                    col_max[j] = vmax

        y = df[LABEL_COL].to_numpy(dtype=np.int64)
        row_positive += int(y.sum())
        if y.max() == 1:
            patient_septic += 1
            onset_hours.append(int(np.argmax(y == 1) + 1))
        if (y[1:] < y[:-1]).any():
            reversion_patients += 1

        iculos = df["ICULOS"].to_numpy(dtype=np.float64)
        if np.any(np.diff(iculos) < 0):
            non_monotonic_iculos += 1

        for k, (lo, hi) in PLAUSIBILITY_RANGES.items():
            vv = df[k].to_numpy(dtype=np.float64)
            mm = ~np.isnan(vv)
            if mm.any():
                obs = vv[mm]
                plaus_counts[k]["observed"] += int(obs.size)
                plaus_counts[k]["outside"] += int(((obs < lo) | (obs > hi)).sum())

    col_names = FEATURE_COLS + [LABEL_COL]
    columns = []
    for i, c in enumerate(col_names):
        miss = 1.0 - (col_non_nan[i] / max(col_total[i], 1))
        columns.append(
            {
                "column": c,
                "missing_frac": float(miss),
                "non_nan": int(col_non_nan[i]),
                "total": int(col_total[i]),
                "min": None if np.isinf(col_min[i]) else float(col_min[i]),
                "max": None if np.isinf(col_max[i]) else float(col_max[i]),
            }
        )

    treatment_cols_present = [
        c for c in EXPECTED_TREATMENT_COLS if c in observed_col_name_set
    ]

    report = {
        "n_patients": len(files),
        "row_total": int(row_total),
        "row_positive": int(row_positive),
        "row_positive_frac": float(row_positive / row_total),
        "patient_positive_count": int(patient_septic),
        "patient_positive_frac": float(patient_septic / len(files)),
        "sequence_length": _safe_stats(np.asarray(lengths, dtype=np.float64)),
        "onset_hour": _safe_stats(np.asarray(onset_hours, dtype=np.float64)),
        "non_monotonic_iculos_patients": int(non_monotonic_iculos),
        "label_reversion_patients": int(reversion_patients),
        "columns": columns,
        "top_missing_columns": sorted(
            columns, key=lambda x: x["missing_frac"], reverse=True
        )[:15],
        "low_missing_columns": sorted(columns, key=lambda x: x["missing_frac"])[:15],
        "plausibility": {
            k: {
                "outside": int(v["outside"]),
                "observed": int(v["observed"]),
                "outside_frac": (
                    float(v["outside"] / v["observed"]) if v["observed"] else None
                ),
            }
            for k, v in plaus_counts.items()
        },
        "treatment_columns_present": treatment_cols_present,
        "has_explicit_treatment_columns": bool(treatment_cols_present),
        "treatment_columns_expected_not_found": [
            c for c in EXPECTED_TREATMENT_COLS if c not in treatment_cols_present
        ],
    }
    return report


def run_qc(
    data_root: Path,
    out_dir: Path,
    train_dirs: list[str],
    max_patients: int | None = None,
) -> Path:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    report = build_quality_report(
        data_root=data_root, train_dirs=train_dirs, max_patients=max_patients
    )
    out_path = qc_dir / "quality_report.json"
    save_json(report, out_path)
    return out_path

