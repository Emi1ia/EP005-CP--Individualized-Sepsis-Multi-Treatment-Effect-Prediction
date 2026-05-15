from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _alloc_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = ["train", "val", "test"]
    raw = {k: float(total) * float(ratios[k]) for k in keys}
    base = {k: int(raw[k]) for k in keys}
    remainder = int(total - sum(base.values()))
    if remainder > 0:
        frac_sorted = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
        for k in frac_sorted[:remainder]:
            base[k] += 1
    return base


def _sample_class_rows(
    frame: pd.DataFrame,
    n: int,
    seed: int,
    allow_replacement: bool,
) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    if frame.empty:
        raise ValueError("Cannot sample from an empty class pool.")
    replace = bool(allow_replacement or (n > len(frame)))
    return frame.sample(n=n, replace=replace, random_state=seed).copy()


def _count_pos(frame: pd.DataFrame, label_col: str) -> int:
    return int((frame[label_col].astype(int) == 1).sum())


def build_balanced_prepared_split(
    source_prepared_dir: Path,
    output_prepared_dir: Path,
    target_positive: int,
    target_negative: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    allow_replacement: bool,
    label_col: str = "septic_patient",
) -> Path:
    if target_positive <= 0 or target_negative <= 0:
        raise ValueError("target_positive and target_negative must be > 0")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    source_splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        split_csv = source_prepared_dir / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing source split file: {split_csv}")
        df = pd.read_csv(split_csv)
        if label_col not in df.columns:
            raise ValueError(f"Missing label column '{label_col}' in {split_csv}")
        source_splits[split] = df

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    pos_targets = _alloc_counts(target_positive, ratios)
    neg_targets = _alloc_counts(target_negative, ratios)

    output_prepared_dir.mkdir(parents=True, exist_ok=True)

    built: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "source_prepared_dir": str(source_prepared_dir),
        "output_prepared_dir": str(output_prepared_dir),
        "target_positive": int(target_positive),
        "target_negative": int(target_negative),
        "target_total": int(target_positive + target_negative),
        "ratios": ratios,
        "allow_replacement": bool(allow_replacement),
        "split_targets": {
            split: {"positive": int(pos_targets[split]), "negative": int(neg_targets[split])}
            for split in ("train", "val", "test")
        },
        "source_split_stats": {},
        "output_split_stats": {},
    }

    for split_i, split in enumerate(("train", "val", "test")):
        src = source_splits[split]
        src_pos = src[src[label_col].astype(int) == 1].copy()
        src_neg = src[src[label_col].astype(int) == 0].copy()

        summary["source_split_stats"][split] = {
            "total": int(len(src)),
            "positive": int(len(src_pos)),
            "negative": int(len(src_neg)),
        }

        n_pos = int(pos_targets[split])
        n_neg = int(neg_targets[split])
        pos_part = _sample_class_rows(src_pos, n=n_pos, seed=seed + 1000 + split_i, allow_replacement=allow_replacement)
        neg_part = _sample_class_rows(src_neg, n=n_neg, seed=seed + 2000 + split_i, allow_replacement=allow_replacement)
        out = pd.concat([pos_part, neg_part], axis=0, ignore_index=True)
        out = out.sample(frac=1.0, random_state=seed + 3000 + split_i).reset_index(drop=True)
        out["source_split"] = split
        out["sample_id"] = [f"{split}_{i:08d}" for i in range(len(out))]

        out_csv = output_prepared_dir / f"{split}.csv"
        out.to_csv(out_csv, index=False)
        built[split] = out

        summary["output_split_stats"][split] = {
            "total": int(len(out)),
            "positive": int(_count_pos(out, label_col)),
            "negative": int(len(out) - _count_pos(out, label_col)),
            "unique_raw_files": int(out["raw_path"].nunique() if "raw_path" in out.columns else 0),
            "unique_prepared_paths": int(out["prepared_path"].nunique() if "prepared_path" in out.columns else 0),
        }

    manifest = pd.concat([built["train"], built["val"], built["test"]], axis=0, ignore_index=True)
    manifest.to_csv(output_prepared_dir / "manifest.csv", index=False)

    summary["output_totals"] = {
        "total": int(len(manifest)),
        "positive": int(_count_pos(manifest, label_col)),
        "negative": int(len(manifest) - _count_pos(manifest, label_col)),
        "unique_raw_files": int(manifest["raw_path"].nunique() if "raw_path" in manifest.columns else 0),
        "unique_prepared_paths": int(
            manifest["prepared_path"].nunique() if "prepared_path" in manifest.columns else 0
        ),
    }

    summary_path = output_prepared_dir / "balanced_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build balanced prepared split CSVs with optional replacement sampling.")
    p.add_argument("--source-prepared-dir", type=Path, required=True)
    p.add_argument("--output-prepared-dir", type=Path, required=True)
    p.add_argument("--target-positive", type=int, default=26202)
    p.add_argument("--target-negative", type=int, default=26202)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-replacement", action="store_true")
    p.add_argument("--label-col", type=str, default="septic_patient")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    summary_path = build_balanced_prepared_split(
        source_prepared_dir=args.source_prepared_dir,
        output_prepared_dir=args.output_prepared_dir,
        target_positive=args.target_positive,
        target_negative=args.target_negative,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        allow_replacement=bool(args.allow_replacement),
        label_col=args.label_col,
    )
    print(f"balanced_split_summary={summary_path}")


if __name__ == "__main__":
    main()
