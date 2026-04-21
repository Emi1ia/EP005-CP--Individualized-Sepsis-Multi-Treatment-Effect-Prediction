from __future__ import annotations

import numpy as np
import torch


def build_temporal_target_torch(
    sepsis_label: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "current",
    horizon_hours: int = 0,
) -> torch.Tensor:
    mode = str(mode).lower()
    y = sepsis_label.float()
    m = mask.bool()

    if mode in {"current", "instant", "timestep"}:
        target = y
    elif mode in {"future_horizon", "horizon", "next_h"}:
        h = max(0, int(horizon_hours))
        target = y.clone()
        if h > 0:
            t = y.size(1)
            for step in range(1, h + 1):
                if step >= t:
                    break
                shifted = torch.zeros_like(y)
                shifted[:, :-step] = y[:, step:]
                target = torch.maximum(target, shifted)
    elif mode in {"patient_any", "patient_level"}:
        any_pos = torch.max(y * m.float(), dim=1, keepdim=True).values
        target = any_pos.expand_as(y)
    else:
        raise ValueError(f"Unknown sepsis target mode: {mode}")

    return target * m.float()


def build_temporal_target_numpy(
    sepsis_label: np.ndarray,
    mode: str = "current",
    horizon_hours: int = 0,
) -> np.ndarray:
    mode = str(mode).lower()
    y = np.asarray(sepsis_label, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D sepsis label array, got shape={y.shape}")

    if mode in {"current", "instant", "timestep"}:
        return y
    if mode in {"future_horizon", "horizon", "next_h"}:
        h = max(0, int(horizon_hours))
        out = y.copy()
        if h > 0:
            n = y.shape[0]
            for step in range(1, h + 1):
                if step >= n:
                    break
                out[:-step] = np.maximum(out[:-step], y[step:])
        return out
    if mode in {"patient_any", "patient_level"}:
        v = float(y.max())
        return np.full_like(y, v, dtype=np.float32)
    raise ValueError(f"Unknown sepsis target mode: {mode}")
