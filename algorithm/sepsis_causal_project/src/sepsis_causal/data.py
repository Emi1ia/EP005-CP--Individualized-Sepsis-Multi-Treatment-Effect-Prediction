from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        max_patients: int | None = None,
        seed: int = 42,
    ):
        self.df = pd.read_csv(split_csv)
        if max_patients is not None and max_patients < len(self.df):
            self.df = (
                self.df.sample(n=int(max_patients), random_state=seed)
                .sort_values("patient_id")
                .reset_index(drop=True)
            )
        if self.df.empty:
            raise ValueError(f"Empty split file: {split_csv}")
        self._cached_patient_sepsis_label: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        payload = np.load(row["prepared_path"])
        x = payload["x"].astype(np.float32)
        actions = payload["actions"].astype(np.int64)
        y = payload["y"].astype(np.float32)
        y_all = payload["y_all"].astype(np.float32)
        sepsis_label = payload["sepsis_label"].astype(np.float32)

        return {
            "x": x,
            "actions": actions,
            "y": y,
            "y_all": y_all,
            "sepsis_label": sepsis_label,
            "length": x.shape[0],
            "patient_id": int(row["patient_id"]),
        }

    def patient_sepsis_label(self) -> np.ndarray:
        if self._cached_patient_sepsis_label is not None:
            return self._cached_patient_sepsis_label

        if "septic_patient" in self.df.columns:
            labels = self.df["septic_patient"].to_numpy(dtype=np.int64, copy=True)
        else:
            labels = np.zeros((len(self.df),), dtype=np.int64)
            for i, p in enumerate(self.df["prepared_path"]):
                payload = np.load(p)
                labels[i] = int(payload["sepsis_label"].max() >= 1.0)
        self._cached_patient_sepsis_label = labels
        return labels

    def balanced_sample_weights(
        self,
        positive_fraction: float = 0.5,
    ) -> np.ndarray:
        labels = self.patient_sepsis_label()
        n = int(labels.size)
        if n == 0:
            raise ValueError("Cannot build sample weights for empty dataset")

        n_pos = int(labels.sum())
        n_neg = int(n - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.ones((n,), dtype=np.float64)

        p = float(np.clip(positive_fraction, 1e-3, 1.0 - 1e-3))
        w_pos = p / n_pos
        w_neg = (1.0 - p) / n_neg
        return np.where(labels == 1, w_pos, w_neg).astype(np.float64, copy=False)


def collate_patient_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_len = max(item["length"] for item in batch)
    x_dim = batch[0]["x"].shape[1]
    num_combos = batch[0]["y_all"].shape[1]

    x = torch.zeros((bsz, max_len, x_dim), dtype=torch.float32)
    actions = torch.zeros((bsz, max_len, 3), dtype=torch.long)
    y = torch.zeros((bsz, max_len), dtype=torch.float32)
    y_all = torch.zeros((bsz, max_len, num_combos), dtype=torch.float32)
    sepsis_label = torch.zeros((bsz, max_len), dtype=torch.float32)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    lengths = torch.zeros((bsz,), dtype=torch.long)
    patient_id = torch.zeros((bsz,), dtype=torch.long)

    for i, item in enumerate(batch):
        t = item["length"]
        x[i, :t] = torch.from_numpy(item["x"])
        actions[i, :t] = torch.from_numpy(item["actions"])
        y[i, :t] = torch.from_numpy(item["y"])
        y_all[i, :t] = torch.from_numpy(item["y_all"])
        sepsis_label[i, :t] = torch.from_numpy(item["sepsis_label"])
        mask[i, :t] = True
        lengths[i] = t
        patient_id[i] = item["patient_id"]

    return {
        "x": x,
        "actions": actions,
        "y": y,
        "y_all": y_all,
        "sepsis_label": sepsis_label,
        "mask": mask,
        "lengths": lengths,
        "patient_id": patient_id,
    }
