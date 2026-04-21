from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import FEATURE_COLS, LABEL_COL
from .utils import parse_patient_id


def list_patient_files(data_root: Path, train_dirs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for d in train_dirs:
        files.extend(sorted((data_root / d).glob("p*.psv")))
    return sorted(files)


def load_patient_frame(path: Path) -> pd.DataFrame:
    cols = FEATURE_COLS + [LABEL_COL]
    return pd.read_csv(path, sep="|", usecols=cols)


def split_patient_files(
    files: list[Path],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[Path]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val : n_train + n_val + n_test]

    out = {
        "train": [files[i] for i in train_idx],
        "val": [files[i] for i in val_idx],
        "test": [files[i] for i in test_idx],
    }
    return out


def files_to_manifest(files: list[Path], split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [parse_patient_id(f) for f in files],
            "split": split_name,
            "raw_path": [str(f) for f in files],
        }
    )

