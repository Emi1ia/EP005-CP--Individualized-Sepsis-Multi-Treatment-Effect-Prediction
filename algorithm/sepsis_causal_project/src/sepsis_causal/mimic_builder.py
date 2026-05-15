from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FEATURE_COLS, LABEL_COL


# Curated MIMIC-IV itemid mapping into the PhysioNet feature schema.
CHART_ITEM_TO_FEATURE = {
    220045: "HR",  # Heart Rate
    220277: "O2Sat",  # O2 saturation pulseoxymetry
    220227: "O2Sat",  # Arterial O2 Saturation
    223762: "Temp",  # Temperature Celsius
    223761: "Temp",  # Temperature Fahrenheit
    220050: "SBP",  # Arterial BP systolic
    220179: "SBP",  # NIBP systolic
    220052: "MAP",  # Arterial BP mean
    220181: "MAP",  # NIBP mean
    220051: "DBP",  # Arterial BP diastolic
    220180: "DBP",  # NIBP diastolic
    220210: "Resp",  # Respiratory Rate
    228640: "EtCO2",  # EtCO2
    229841: "FiO2",  # FiO2 (CH)
    229280: "FiO2",  # FiO2 (ECMO)
}

LAB_ITEM_TO_FEATURE = {
    50820: "pH",
    50818: "PaCO2",
    50817: "SaO2",
    50878: "AST",
    51006: "BUN",  # Urea Nitrogen
    51842: "BUN",  # Bun
    50863: "Alkalinephos",
    50893: "Calcium",
    50902: "Chloride",
    50912: "Creatinine",
    50883: "Bilirubin_direct",
    50885: "Bilirubin_total",
    50931: "Glucose",
    50809: "Glucose",
    50813: "Lactate",
    50960: "Magnesium",
    50970: "Phosphate",
    50971: "Potassium",
    50833: "Potassium",
    51002: "TroponinI",
    51003: "TroponinI",
    51221: "Hct",
    50810: "Hct",
    51222: "Hgb",
    50811: "Hgb",
    51275: "PTT",
    51300: "WBC",
    51301: "WBC",
    52407: "WBC",
    51214: "Fibrinogen",
    51623: "Fibrinogen",
    51265: "Platelets",
    51704: "Platelets",
}


SEPSIS_ICD9_PREFIX = ("038", "99591", "99592", "78552")
SEPSIS_ICD10_PREFIX = ("A40", "A41", "R6520", "R6521", "R652")

ANTIBIOTIC_KEYWORDS = (
    "cillin",
    "cef",
    "meropenem",
    "imipenem",
    "ertapenem",
    "vancomycin",
    "linezolid",
    "daptomycin",
    "aztreonam",
    "clindamycin",
    "metronidazole",
    "ciprofloxacin",
    "levofloxacin",
    "moxifloxacin",
    "gentamicin",
    "amikacin",
    "tobramycin",
    "trimethoprim",
    "sulfamethoxazole",
    "doxycycline",
    "tigecycline",
    "colistin",
)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _normalize_value(feature: str, itemid: int, values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if feature == "Temp" and itemid == 223761:
        # Convert Fahrenheit to Celsius.
        out = (out - 32.0) / 1.8
    if feature == "FiO2":
        # FiO2 in MIMIC is often in percentage points.
        out = np.where(out > 1.0, out / 100.0, out)
        out = pd.Series(out, index=values.index)
    return out


def _extract_sepsis_hadm_ids(dx: pd.DataFrame) -> set[int]:
    code = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    v9 = (
        (dx["icd_version"] == 9)
        & (
            code.str.startswith(SEPSIS_ICD9_PREFIX[0])
            | code.str.startswith(SEPSIS_ICD9_PREFIX[1])
            | code.str.startswith(SEPSIS_ICD9_PREFIX[2])
            | code.str.startswith(SEPSIS_ICD9_PREFIX[3])
        )
    )
    v10 = (
        (dx["icd_version"] == 10)
        & (
            code.str.startswith(SEPSIS_ICD10_PREFIX[0])
            | code.str.startswith(SEPSIS_ICD10_PREFIX[1])
            | code.str.startswith(SEPSIS_ICD10_PREFIX[2])
            | code.str.startswith(SEPSIS_ICD10_PREFIX[3])
            | code.str.startswith(SEPSIS_ICD10_PREFIX[4])
        )
    )
    return set(dx.loc[v9 | v10, "hadm_id"].dropna().astype(int).tolist())


def _build_onset_time_map(
    mimic_root: Path,
) -> dict[int, pd.Timestamp]:
    hosp_dir = mimic_root / "hosp"

    onset_candidates: list[pd.DataFrame] = []

    emar_path = hosp_dir / "emar.csv.gz"
    if emar_path.exists():
        emar = pd.read_csv(emar_path, usecols=["hadm_id", "charttime", "medication"])
        med = emar["medication"].astype(str).str.lower()
        mask_abx = np.zeros((len(emar),), dtype=bool)
        for kw in ANTIBIOTIC_KEYWORDS:
            mask_abx |= med.str.contains(kw, regex=False, na=False).to_numpy()
        emar = emar.loc[mask_abx].copy()
        emar["event_time"] = _to_datetime(emar["charttime"])
        emar = emar.dropna(subset=["hadm_id", "event_time"])
        onset_candidates.append(
            emar.groupby("hadm_id", as_index=False)["event_time"].min()
        )

    micro_path = hosp_dir / "microbiologyevents.csv.gz"
    if micro_path.exists():
        micro = pd.read_csv(micro_path, usecols=["hadm_id", "charttime", "chartdate"])
        t = _to_datetime(micro["charttime"])
        d = _to_datetime(micro["chartdate"])
        micro["event_time"] = t.fillna(d)
        micro = micro.dropna(subset=["hadm_id", "event_time"])
        onset_candidates.append(
            micro.groupby("hadm_id", as_index=False)["event_time"].min()
        )

    if not onset_candidates:
        return {}

    merged = pd.concat(onset_candidates, ignore_index=True)
    merged = merged.groupby("hadm_id", as_index=False)["event_time"].min()
    return {
        int(row.hadm_id): pd.Timestamp(row.event_time)
        for row in merged.itertuples(index=False)
    }


def _event_groups_chart(mimic_root: Path) -> dict[int, pd.DataFrame]:
    icu_dir = mimic_root / "icu"
    ce = pd.read_csv(
        icu_dir / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
    )
    ce = ce.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"]).copy()
    ce["itemid"] = ce["itemid"].astype(int)
    ce = ce.loc[ce["itemid"].isin(CHART_ITEM_TO_FEATURE.keys())].copy()
    ce["event_time"] = _to_datetime(ce["charttime"])
    ce = ce.dropna(subset=["event_time"])
    ce["feature"] = ce["itemid"].map(CHART_ITEM_TO_FEATURE)
    ce["value"] = np.nan
    for itemid, feature in CHART_ITEM_TO_FEATURE.items():
        m = ce["itemid"] == itemid
        if m.any():
            ce.loc[m, "value"] = _normalize_value(feature, itemid, ce.loc[m, "valuenum"])
    ce = ce.dropna(subset=["value"])
    ce = ce[["stay_id", "event_time", "feature", "value"]]
    return {
        int(stay_id): g[["event_time", "feature", "value"]].copy()
        for stay_id, g in ce.groupby("stay_id")
    }


def _event_groups_lab(mimic_root: Path) -> dict[int, pd.DataFrame]:
    hosp_dir = mimic_root / "hosp"
    le = pd.read_csv(
        hosp_dir / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
    )
    le = le.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"]).copy()
    le["itemid"] = le["itemid"].astype(int)
    le = le.loc[le["itemid"].isin(LAB_ITEM_TO_FEATURE.keys())].copy()
    le["event_time"] = _to_datetime(le["charttime"])
    le = le.dropna(subset=["event_time"])
    le["feature"] = le["itemid"].map(LAB_ITEM_TO_FEATURE)
    le["value"] = pd.to_numeric(le["valuenum"], errors="coerce")
    le = le.dropna(subset=["value"])
    le = le[["hadm_id", "event_time", "feature", "value"]]
    return {
        int(hadm_id): g[["event_time", "feature", "value"]].copy()
        for hadm_id, g in le.groupby("hadm_id")
    }


def _add_events_to_frame(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    intime: pd.Timestamp,
    outtime: pd.Timestamp,
) -> None:
    if events.empty:
        return
    w = events[(events["event_time"] >= intime) & (events["event_time"] <= outtime)].copy()
    if w.empty:
        return
    hours = np.floor((w["event_time"] - intime).dt.total_seconds() / 3600.0).astype(int) + 1
    w["ICULOS"] = hours
    w = w[(w["ICULOS"] >= 1) & (w["ICULOS"] <= len(frame))]
    if w.empty:
        return

    agg = w.groupby(["ICULOS", "feature"], as_index=False)["value"].mean()
    for row in agg.itertuples(index=False):
        frame.at[int(row.ICULOS) - 1, str(row.feature)] = float(row.value)


def build_mimic_training_set(
    mimic_root: Path,
    output_dir: Path,
    min_hours: int = 8,
    patient_id_offset: int = 9_000_000,
) -> dict[str, Any]:
    mimic_root = mimic_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    icu_dir = mimic_root / "icu"
    hosp_dir = mimic_root / "hosp"
    required = [
        icu_dir / "icustays.csv.gz",
        icu_dir / "chartevents.csv.gz",
        hosp_dir / "labevents.csv.gz",
        hosp_dir / "patients.csv.gz",
        hosp_dir / "admissions.csv.gz",
        hosp_dir / "diagnoses_icd.csv.gz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required MIMIC files: {missing}")

    stays = pd.read_csv(icu_dir / "icustays.csv.gz")
    stays["intime"] = _to_datetime(stays["intime"])
    stays["outtime"] = _to_datetime(stays["outtime"])
    stays = stays.dropna(subset=["stay_id", "hadm_id", "subject_id", "intime", "outtime"]).copy()
    stays["duration_h"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    stays = stays[stays["duration_h"] >= float(min_hours)].copy()

    patients = pd.read_csv(hosp_dir / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(hosp_dir / "admissions.csv.gz", usecols=["hadm_id", "admittime"])
    admissions["admittime"] = _to_datetime(admissions["admittime"])
    dx = pd.read_csv(hosp_dir / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"])

    sepsis_hadm = _extract_sepsis_hadm_ids(dx)
    onset_map = _build_onset_time_map(mimic_root)
    chart_groups = _event_groups_chart(mimic_root)
    lab_groups = _event_groups_lab(mimic_root)

    demo = stays.merge(patients, on="subject_id", how="left").merge(admissions, on="hadm_id", how="left")

    manifest_rows: list[dict[str, Any]] = []
    written = 0
    septic_stays = 0

    for i, row in enumerate(demo.itertuples(index=False), 1):
        stay_id = int(row.stay_id)
        hadm_id = int(row.hadm_id)
        intime = pd.Timestamp(row.intime)
        outtime = pd.Timestamp(row.outtime)
        t = int(np.ceil((outtime - intime).total_seconds() / 3600.0))
        if t < min_hours:
            continue

        frame = pd.DataFrame(index=np.arange(t), columns=FEATURE_COLS, dtype=np.float64)
        frame["ICULOS"] = np.arange(1, t + 1, dtype=np.int64)
        frame["Age"] = float(row.anchor_age) if pd.notna(row.anchor_age) else np.nan
        g = str(row.gender).strip().upper() if pd.notna(row.gender) else ""
        frame["Gender"] = 1.0 if g == "M" else (0.0 if g == "F" else np.nan)
        frame["Unit1"] = np.nan
        frame["Unit2"] = np.nan
        if pd.notna(row.admittime):
            frame["HospAdmTime"] = float((pd.Timestamp(row.admittime) - intime).total_seconds() / 3600.0)
        else:
            frame["HospAdmTime"] = np.nan

        if stay_id in chart_groups:
            _add_events_to_frame(frame, chart_groups[stay_id], intime, outtime)
        if hadm_id in lab_groups:
            _add_events_to_frame(frame, lab_groups[hadm_id], intime, outtime)

        # Build a monotonic per-hour sepsis label.
        sepsis = np.zeros((t,), dtype=np.int64)
        is_septic = hadm_id in sepsis_hadm
        if is_septic:
            septic_stays += 1
            onset_time = onset_map.get(hadm_id, intime)
            onset_h = int(np.floor((onset_time - intime).total_seconds() / 3600.0)) + 1
            onset_h = int(np.clip(onset_h, 1, t))
            sepsis[onset_h - 1 :] = 1
        frame[LABEL_COL] = sepsis

        pid = patient_id_offset + written + 1
        out_path = output_dir / f"p{pid:07d}.psv"
        frame.to_csv(out_path, sep="|", index=False)
        written += 1

        manifest_rows.append(
            {
                "patient_id": pid,
                "mimic_stay_id": stay_id,
                "mimic_hadm_id": hadm_id,
                "subject_id": int(row.subject_id),
                "length": t,
                "septic_patient": int(is_septic),
                "path": str(out_path),
            }
        )

        if i % 100 == 0:
            print(f"mimic build: {i}/{len(demo)} stays processed, written={written}")

    manifest_path = output_dir / "mimic_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "mimic_root": str(mimic_root),
        "output_dir": str(output_dir),
        "min_hours": int(min_hours),
        "patient_id_offset": int(patient_id_offset),
        "written_patients": int(written),
        "septic_patients": int(septic_stays),
        "septic_patient_frac": float(septic_stays / written) if written else None,
        "manifest_path": str(manifest_path),
    }
    (output_dir / "mimic_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
