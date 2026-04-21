from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .evaluate import run_evaluate
from .optimize import run_optimize
from .prepare import run_prepare
from .qc import run_qc
from .tune import run_tuning
from .train import run_train
from .utils import load_config, resolve_run_paths, set_global_seed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sepsis causal transformer pipeline")
    p.add_argument("--config", type=str, default="configs/base/default.yaml")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-patients", type=int, default=None)

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("qc")
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    sub.add_parser("tune")
    sub.add_parser("optimize")
    sub.add_parser("full-run")
    return p


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.data_root is not None:
        config["paths"]["data_root"] = args.data_root
    if args.out_dir is not None:
        config["paths"]["out_dir"] = args.out_dir
    if args.max_patients is not None:
        config["data"]["max_patients"] = args.max_patients
    return config


def _print_paths(data_root: Path, out_dir: Path) -> None:
    print(f"data_root={data_root}")
    print(f"out_dir={out_dir}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = _apply_overrides(config, args)
    set_global_seed(int(config["seed"]))

    data_root, out_dir = resolve_run_paths(config)
    _print_paths(data_root, out_dir)

    train_dirs = config["data"]["train_dirs"]
    max_patients = config["data"]["max_patients"]

    if args.command == "qc":
        qc_path = run_qc(
            data_root=data_root,
            out_dir=out_dir,
            train_dirs=train_dirs,
            max_patients=max_patients,
        )
        print(f"qc_report={qc_path}")
        return

    if args.command == "prepare":
        prep_dir = run_prepare(config=config, data_root=data_root, out_dir=out_dir)
        print(f"prepared_dir={prep_dir}")
        return

    if args.command == "train":
        model_path = run_train(config=config, out_dir=out_dir)
        print(f"best_model={model_path}")
        return

    if args.command == "evaluate":
        metrics_path = run_evaluate(config=config, out_dir=out_dir)
        print(f"metrics={metrics_path}")
        return

    if args.command == "tune":
        result_path = run_tuning(config=config, out_dir=out_dir)
        print(f"tuning_result={result_path}")
        return

    if args.command == "optimize":
        optimized_config_path = run_optimize(config=config, data_root=data_root, out_dir=out_dir)
        print(f"optimized_config={optimized_config_path}")
        return

    if args.command == "full-run":
        qc_path = run_qc(
            data_root=data_root,
            out_dir=out_dir,
            train_dirs=train_dirs,
            max_patients=max_patients,
        )
        print(f"qc_report={qc_path}")
        prep_dir = run_prepare(config=config, data_root=data_root, out_dir=out_dir)
        print(f"prepared_dir={prep_dir}")
        model_path = run_train(config=config, out_dir=out_dir)
        print(f"best_model={model_path}")
        metrics_path = run_evaluate(config=config, out_dir=out_dir)
        print(f"metrics={metrics_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
