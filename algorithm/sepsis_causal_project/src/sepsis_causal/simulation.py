from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ACTION_COMBOS, NUM_ACTION_COMBOS


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass
class SimConfig:
    noise_std: float = 0.15


def _abnormal_high(x: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    return np.maximum(0.0, (x - threshold) / scale)


def _abnormal_low(x: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    return np.maximum(0.0, (threshold - x) / scale)


def derive_severity_terms(raw_values: np.ndarray, col_idx: dict[str, int]) -> dict[str, np.ndarray]:
    hr = raw_values[:, col_idx["HR"]]
    map_ = raw_values[:, col_idx["MAP"]]
    sbp = raw_values[:, col_idx["SBP"]]
    resp = raw_values[:, col_idx["Resp"]]
    temp = raw_values[:, col_idx["Temp"]]
    lact = raw_values[:, col_idx["Lactate"]]
    creat = raw_values[:, col_idx["Creatinine"]]
    o2sat = raw_values[:, col_idx["O2Sat"]]
    wbc = raw_values[:, col_idx["WBC"]]
    age = raw_values[:, col_idx["Age"]]

    hr_term = _abnormal_high(hr, 100.0, 35.0)
    map_term = _abnormal_low(map_, 65.0, 20.0)
    sbp_term = _abnormal_low(sbp, 100.0, 35.0)
    resp_term = _abnormal_high(resp, 22.0, 10.0)
    temp_high = _abnormal_high(temp, 38.3, 1.2)
    temp_low = _abnormal_low(temp, 36.0, 1.2)
    lact_term = _abnormal_high(lact, 2.0, 3.0)
    creat_term = _abnormal_high(creat, 1.2, 2.0)
    o2_term = _abnormal_low(o2sat, 92.0, 15.0)
    wbc_term = np.maximum(0.0, (np.abs(wbc - 8.0) - 4.0) / 10.0)
    age_term = np.maximum(0.0, (age - 60.0) / 35.0)

    infection_load = 0.60 * temp_high + 0.40 * wbc_term + 0.25 * lact_term
    shock_load = 0.55 * map_term + 0.35 * lact_term + 0.25 * o2_term
    fluid_sensitive = 0.45 * o2_term + 0.20 * age_term

    severity = (
        0.40 * hr_term
        + 0.80 * map_term
        + 0.35 * sbp_term
        + 0.45 * resp_term
        + 0.25 * temp_high
        + 0.25 * temp_low
        + 0.50 * lact_term
        + 0.30 * creat_term
        + 0.25 * o2_term
        + 0.20 * age_term
    )

    return {
        "severity": severity,
        "infection_load": infection_load,
        "shock_load": shock_load,
        "fluid_sensitive": fluid_sensitive,
    }


def sample_actions(
    terms: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    sev = terms["severity"]
    inf = terms["infection_load"]
    shock = terms["shock_load"]

    p_abx = sigmoid(-1.0 + 0.70 * sev + 0.60 * inf + rng.normal(0.0, 0.1, size=sev.shape))
    abx = (rng.random(size=p_abx.shape) < p_abx).astype(np.int64)

    fluid_logits = np.stack(
        [
            1.20 - 1.00 * shock - 0.40 * sev,
            0.25 + 0.80 * shock + 0.25 * sev,
            -0.80 + 1.40 * shock + 0.60 * sev,
        ],
        axis=-1,
    )
    fluid_prob = softmax(fluid_logits, axis=-1)
    fluid = np.array([rng.choice(3, p=p) for p in fluid_prob], dtype=np.int64)

    vaso_logits = np.stack(
        [
            1.10 - 1.10 * shock - 0.40 * sev,
            0.10 + 0.90 * shock + 0.35 * sev,
            -0.95 + 1.70 * shock + 0.70 * sev,
        ],
        axis=-1,
    )
    vaso_prob = softmax(vaso_logits, axis=-1)
    vaso = np.array([rng.choice(3, p=p) for p in vaso_prob], dtype=np.int64)

    actions = np.stack([abx, fluid, vaso], axis=-1)
    return actions


def generate_potential_outcomes(
    terms: dict[str, np.ndarray],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sev = terms["severity"]
    inf = terms["infection_load"]
    shock = terms["shock_load"]
    fluid_sensitive = terms["fluid_sensitive"]

    base = -2.20 + 1.20 * sev + 0.45 * shock

    # Shape [T, 18]
    y_all = np.zeros((sev.shape[0], NUM_ACTION_COMBOS), dtype=np.float32)
    for combo_idx, (abx, fluid, vaso) in enumerate(ACTION_COMBOS):
        abx_eff = abx * (0.75 * inf - 0.10 * (1.0 - inf))
        fluid_eff = fluid * (0.32 * shock - 0.20 * fluid_sensitive)
        vaso_eff = vaso * (0.45 * shock - 0.12 * (1.0 - shock))
        interaction = 0.08 * abx * fluid + 0.10 * fluid * vaso + 0.06 * abx * vaso
        logit = base - abx_eff - fluid_eff - vaso_eff - interaction
        logit = logit + rng.normal(0.0, noise_std, size=sev.shape)
        y_all[:, combo_idx] = sigmoid(logit).astype(np.float32)
    return y_all


def factual_from_actions(y_all: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    combo_idx = actions[:, 0] * 9 + actions[:, 1] * 3 + actions[:, 2]
    factual_prob = y_all[np.arange(y_all.shape[0]), combo_idx]
    y = (rng.random(size=factual_prob.shape) < factual_prob).astype(np.float32)
    return y

