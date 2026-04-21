from __future__ import annotations

from itertools import product


LABEL_COL = "SepsisLabel"

FEATURE_COLS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

ACTION_DIMS = {
    "abx": 2,
    "fluid": 3,
    "vaso": 3,
}

ACTION_COMBOS = list(product(range(2), range(3), range(3)))
NUM_ACTION_COMBOS = len(ACTION_COMBOS)

