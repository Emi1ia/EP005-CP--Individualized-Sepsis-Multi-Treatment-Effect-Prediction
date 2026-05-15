from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_benchmark_rows(benchmark_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    rows = payload.get("results_ranked", [])
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "model": str(row.get("model_type", "unknown")),
                "source": str(benchmark_json),
                "sepsis_auroc": _to_float(row.get("sepsis_auroc")),
                "sepsis_auprc": _to_float(row.get("sepsis_auprc")),
                "patient_auroc": _to_float(row.get("patient_sepsis_auroc")),
                "patient_auprc": _to_float(row.get("patient_sepsis_auprc")),
                "pehe": _to_float(row.get("pehe")),
                "ate_error": _to_float(row.get("ate_error")),
                "policy_regret": _to_float(row.get("policy_regret")),
            }
        )
    return out


def _load_single_metrics_row(metrics_json: Path, model_label: str) -> dict[str, Any]:
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    sepsis = payload.get("sepsis_metrics", {})
    patient = payload.get("patient_sepsis_metrics", {})
    treat = payload.get("treatment_effect_metrics", {})
    return {
        "model": model_label,
        "source": str(metrics_json),
        "sepsis_auroc": _to_float(sepsis.get("auroc")),
        "sepsis_auprc": _to_float(sepsis.get("auprc")),
        "patient_auroc": _to_float(patient.get("auroc")),
        "patient_auprc": _to_float(patient.get("auprc")),
        "pehe": _to_float(treat.get("pehe")),
        "ate_error": _to_float(treat.get("ate_error")),
        "policy_regret": _to_float(treat.get("policy_regret")),
    }


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    pehe = row["pehe"] if row["pehe"] is not None else 1e9
    ate = row["ate_error"] if row["ate_error"] is not None else 1e9
    regret = row["policy_regret"] if row["policy_regret"] is not None else 1e9
    return pehe, ate, regret


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank_treatment",
        "model",
        "sepsis_auroc",
        "sepsis_auprc",
        "patient_auroc",
        "patient_auprc",
        "pehe",
        "ate_error",
        "policy_regret",
        "source",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            rec = dict(row)
            rec["rank_treatment"] = i
            w.writerow(rec)


def _write_markdown(rows: list[dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Treatment Recommendation Comparison",
        "",
        "Ranked by treatment metrics priority: `PEHE asc`, `ATE error asc`, `policy_regret asc`.",
        "",
        "| Rank | Model | Sepsis AUROC | Sepsis AUPRC | Patient AUROC | Patient AUPRC | PEHE | ATE Error | Policy Regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    row["model"],
                    _fmt(row["sepsis_auroc"]),
                    _fmt(row["sepsis_auprc"]),
                    _fmt(row["patient_auroc"]),
                    _fmt(row["patient_auprc"]),
                    _fmt(row["pehe"]),
                    _fmt(row["ate_error"]),
                    _fmt(row["policy_regret"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Sources", ""])
    for row in rows:
        lines.append(f"- `{row['model']}`: `{row['source']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build treatment-focused model comparison table.")
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--extra-metrics-json", type=Path, default=None)
    parser.add_argument("--extra-label", type=str, default="transformer_tuned")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    if not args.benchmark_json.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark_json}")

    rows = _load_benchmark_rows(args.benchmark_json)
    if args.extra_metrics_json is not None:
        if not args.extra_metrics_json.exists():
            raise FileNotFoundError(f"Extra metrics file not found: {args.extra_metrics_json}")
        rows.append(_load_single_metrics_row(args.extra_metrics_json, args.extra_label))

    rows_sorted = sorted(rows, key=_sort_key)
    _write_csv(rows_sorted, args.out_csv)
    _write_markdown(rows_sorted, args.out_md)
    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_md={args.out_md}")
    print(f"rows={len(rows_sorted)}")


if __name__ == "__main__":
    main()
