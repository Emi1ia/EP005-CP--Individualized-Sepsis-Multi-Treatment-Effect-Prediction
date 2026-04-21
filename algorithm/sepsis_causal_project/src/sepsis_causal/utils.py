from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(payload: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_run_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    out_dir = Path(config["paths"]["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_root, out_dir


def parse_patient_id(path: Path) -> int:
    stem = path.stem
    return int(stem[1:])

