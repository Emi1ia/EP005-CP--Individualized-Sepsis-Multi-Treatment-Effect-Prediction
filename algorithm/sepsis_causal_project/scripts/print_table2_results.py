from __future__ import annotations

import argparse
import json
from pathlib import Path

def _repo_root() -> Path:
    # script path: algorithm/sepsis_causal_project/scripts/print_table2_results.py
    return Path(__file__).resolve().parents[3]


def _require(d: dict, key: str):
    if key not in d:
        raise KeyError(f"Missing key '{key}' in metrics payload.")
    return d[key]


def _load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in metrics file {metrics_path}: {e}") from e


def _to_float(v) -> float:
    return float(v)


def _extract_table2_fields(metrics: dict, percent: bool) -> dict:
    sepsis_metrics = _require(metrics, "sepsis_metrics")
    patient_metrics = _require(metrics, "patient_sepsis_metrics")
    sepsis_error = _require(metrics, "sepsis_error_metrics")
    treatment = _require(metrics, "treatment_effect_metrics")

    auroc_s = _to_float(_require(sepsis_metrics, "auroc"))
    auroc_p = _to_float(_require(patient_metrics, "auroc"))
    auprc = _to_float(_require(patient_metrics, "auprc"))
    f1 = _to_float(_require(patient_metrics, "f1"))
    mae = _to_float(_require(sepsis_error, "mae"))
    rmse = _to_float(_require(sepsis_error, "rmse"))
    pehe = _to_float(_require(treatment, "pehe"))
    ate_error = _to_float(_require(treatment, "ate_error"))

    if percent:
        auroc_s *= 100.0
        auroc_p *= 100.0
        auprc *= 100.0
        f1 *= 100.0

    return {
        "model_type": "Causal Transformer",
        "AUROC_S": auroc_s,
        "AUROC_P": auroc_p,
        "AUPRC": auprc,
        "F1": f1,
        "MAE": mae,
        "RMSE": rmse,
        "PEHE": pehe,
        "ATE_Error": ate_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output numeric Table-2 style values from real eval/metrics.json."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to eval metrics JSON produced by real training/evaluation.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "plain"],
        help="Output numeric format.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Scale AUROC/AUPRC/F1 from [0,1] to [0,100].",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output file path for numeric output.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics).resolve()
    metrics = _load_metrics(metrics_path)
    payload = _extract_table2_fields(metrics, percent=bool(args.percent))

    if args.format == "json":
        output_text = json.dumps(payload, indent=2)
        print(output_text)
    else:
        output_text = (
            f"AUROC_S={payload['AUROC_S']}\n"
            f"AUROC_P={payload['AUROC_P']}\n"
            f"AUPRC={payload['AUPRC']}\n"
            f"F1={payload['F1']}\n"
            f"MAE={payload['MAE']}\n"
            f"RMSE={payload['RMSE']}\n"
            f"PEHE={payload['PEHE']}\n"
            f"ATE_Error={payload['ATE_Error']}"
        )
        print(output_text)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
