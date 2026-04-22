from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import FEATURE_COLS


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        max_patients: int | None = None,
        seed: int = 42,
        augment: bool = False,
        augment_cfg: dict[str, Any] | None = None,
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
        self.augment = bool(augment)
        cfg = augment_cfg or {}
        self.aug_positive_only = bool(cfg.get("positive_only", True))
        self.aug_apply_prob = float(np.clip(cfg.get("apply_prob", 1.0), 0.0, 1.0))
        self.aug_noise_std = float(max(0.0, cfg.get("noise_std", 0.0)))
        self.aug_scale_std = float(max(0.0, cfg.get("scale_std", 0.0)))
        self.aug_feature_dropout_prob = float(np.clip(cfg.get("feature_dropout_prob", 0.0), 0.0, 1.0))
        self.aug_time_dropout_prob = float(np.clip(cfg.get("time_dropout_prob", 0.0), 0.0, 1.0))
        raw_value_dim = cfg.get("value_dim", len(FEATURE_COLS))
        if raw_value_dim is None:
            raw_value_dim = len(FEATURE_COLS)
        self.aug_value_dim = int(max(1, raw_value_dim))
        raw_clip = cfg.get("value_clip", 8.0)
        self.aug_value_clip = None if raw_clip is None else float(max(0.0, raw_clip))
        self._rng = np.random.default_rng(seed + 7919)

    def __len__(self) -> int:
        return len(self.df)

    def _augment_x(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        value_dim = int(min(max(1, self.aug_value_dim), out.shape[1]))
        xv = out[:, :value_dim]

        if self.aug_scale_std > 0:
            scale = self._rng.normal(loc=1.0, scale=self.aug_scale_std, size=(1, value_dim)).astype(np.float32)
            xv *= scale

        if self.aug_noise_std > 0:
            xv += self._rng.normal(loc=0.0, scale=self.aug_noise_std, size=xv.shape).astype(np.float32)

        if self.aug_feature_dropout_prob > 0:
            drop_mask = self._rng.random(size=xv.shape) < self.aug_feature_dropout_prob
            xv[drop_mask] = 0.0

        if self.aug_time_dropout_prob > 0:
            t_drop = self._rng.random(size=(xv.shape[0],)) < self.aug_time_dropout_prob
            xv[t_drop, :] = 0.0

        if self.aug_value_clip is not None and self.aug_value_clip > 0:
            np.clip(xv, -self.aug_value_clip, self.aug_value_clip, out=xv)

        out[:, :value_dim] = xv
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        payload = np.load(row["prepared_path"])
        x = payload["x"].astype(np.float32)
        actions = payload["actions"].astype(np.int64)
        y = payload["y"].astype(np.float32)
        y_all = payload["y_all"].astype(np.float32)
        sepsis_label = payload["sepsis_label"].astype(np.float32)
        if self.augment and (self._rng.random() < self.aug_apply_prob):
            is_positive = bool(np.max(sepsis_label) >= 1.0)
            if (not self.aug_positive_only) or is_positive:
                x = self._augment_x(x)

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
