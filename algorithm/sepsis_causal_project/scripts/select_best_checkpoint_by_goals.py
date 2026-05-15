from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Goal:
    name: str
    key: str
    direction: str  # "max" or "min"
    min_value: float | None
    max_value: float | None
    weight: float = 1.0


def _json_loads_maybe(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _suggest_tune_root(out_dir: Path, study_name: str) -> Path:
    candidate = out_dir / "tuning" / study_name
    if candidate.exists():
        return candidate
    return out_dir / "tuning" / study_name


def _load_trials_from_optuna_db(db_path: Path, study_name: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    study_row = cur.fetchone()
    if study_row is None:
        raise ValueError(f"Study '{study_name}' not found in {db_path}")
    study_id = int(study_row[0])

    cur.execute(
        """
        SELECT t.trial_id, t.number, t.state, tv.value
        FROM trials t
        LEFT JOIN trial_values tv
          ON tv.trial_id = t.trial_id AND tv.objective = 0
        WHERE t.study_id = ?
        ORDER BY t.number
        """,
        (study_id,),
    )
    trial_rows = cur.fetchall()

    trial_ids = [int(r[0]) for r in trial_rows]
    if not trial_ids:
        return []

    placeholders = ",".join(["?"] * len(trial_ids))
    cur.execute(
        f"""
        SELECT trial_id, key, value_json
        FROM trial_user_attributes
        WHERE trial_id IN ({placeholders})
        """,
        tuple(trial_ids),
    )
    attr_rows = cur.fetchall()

    cur.execute(
        f"""
        SELECT trial_id, param_name, param_value, distribution_json
        FROM trial_params
        WHERE trial_id IN ({placeholders})
        """,
        tuple(trial_ids),
    )
    param_rows = cur.fetchall()

    attrs_by_trial: dict[int, dict[str, Any]] = {}
    for tid_raw, key_raw, value_raw in attr_rows:
        tid = int(tid_raw)
        attrs_by_trial.setdefault(tid, {})[str(key_raw)] = _json_loads_maybe(value_raw)

    params_by_trial: dict[int, dict[str, Any]] = {}
    for tid_raw, name_raw, value_raw, dist_raw in param_rows:
        tid = int(tid_raw)
        name = str(name_raw)
        val = float(value_raw)
        out_val: Any = val

        dist = _json_loads_maybe(dist_raw)
        if isinstance(dist, dict) and dist.get("name") == "CategoricalDistribution":
            choices = dist.get("attributes", {}).get("choices", [])
            idx = int(round(val))
            if 0 <= idx < len(choices):
                out_val = choices[idx]
        elif abs(val - round(val)) < 1e-10 and (
            name.endswith("hidden_size")
            or name.endswith("num_heads")
            or name.endswith("num_layers")
            or name.endswith("ff_multiplier")
            or name.endswith("batch_size")
        ):
            out_val = int(round(val))

        params_by_trial.setdefault(tid, {})[name] = out_val

    trials: list[dict[str, Any]] = []
    for trial_id_raw, number_raw, state_raw, value_raw in trial_rows:
        tid = int(trial_id_raw)
        trials.append(
            {
                "trial_id": tid,
                "number": int(number_raw),
                "state": str(state_raw),
                "objective_value": (None if value_raw is None else float(value_raw)),
                "user_attrs": attrs_by_trial.get(tid, {}),
                "params": params_by_trial.get(tid, {}),
            }
        )
    return trials


def _goal_span(goal: Goal) -> float:
    if goal.min_value is not None and goal.max_value is not None and goal.max_value > goal.min_value:
        return goal.max_value - goal.min_value
    anchor = goal.max_value if goal.direction == "min" else goal.min_value
    if anchor is None or abs(anchor) < 1e-8:
        return 1.0
    return abs(anchor)


def _score_goal(
    value: float | None,
    goal: Goal,
    strict_lower_bounds_for_min: bool,
) -> tuple[bool, float, float]:
    """Return (met, violation, utility). Lower violation is better; higher utility is better."""
    if value is None:
        return False, 1e6, -1e6

    span = _goal_span(goal)
    violation = 0.0

    if goal.direction == "max":
        if goal.min_value is not None and value < goal.min_value:
            violation += (goal.min_value - value) / span
        if goal.max_value is not None and value > goal.max_value:
            violation += (value - goal.max_value) / span
    else:
        if goal.max_value is not None and value > goal.max_value:
            violation += (value - goal.max_value) / span
        if strict_lower_bounds_for_min and goal.min_value is not None and value < goal.min_value:
            violation += (goal.min_value - value) / span

    met = violation <= 1e-12

    if goal.direction == "max":
        anchor = goal.min_value if goal.min_value is not None else 0.0
        utility = (value - anchor) / span
    else:
        anchor = goal.max_value if goal.max_value is not None else 1.0
        utility = (anchor - value) / span
    w = float(max(0.0, goal.weight))
    return met, float(violation * w), float(utility * w)


def _metric_value(attrs: dict[str, Any], key: str) -> float | None:
    raw = attrs.get(key, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_goals_from_yaml(path: Path) -> list[Goal]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--goals-yaml must contain a list of goal objects.")

    goals: list[Goal] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"goals[{i}] must be an object.")
        name = str(item.get("name", f"goal_{i}"))
        key = str(item.get("key", "")).strip()
        direction = str(item.get("direction", "")).lower().strip()
        if not key:
            raise ValueError(f"goals[{i}].key is required.")
        if direction not in {"max", "min"}:
            raise ValueError(f"goals[{i}].direction must be 'max' or 'min'.")
        min_value = item.get("min", None)
        max_value = item.get("max", None)
        weight = float(item.get("weight", 1.0))
        goals.append(
            Goal(
                name=name,
                key=key,
                direction=direction,
                min_value=(None if min_value is None else float(min_value)),
                max_value=(None if max_value is None else float(max_value)),
                weight=weight,
            )
        )
    return goals


def _build_default_goals(args: argparse.Namespace) -> list[Goal]:
    if args.classification_level == "patient":
        cls_prefix = "val_patient_sepsis"
        err_prefix = "val_patient_sepsis"
    else:
        cls_prefix = "val_sepsis"
        err_prefix = "val_sepsis"

    return [
        Goal("classification_auroc", f"{cls_prefix}_auroc", "max", args.auroc_min, None),
        Goal("classification_auprc", f"{cls_prefix}_auprc", "max", args.auprc_min, args.auprc_max),
        Goal("classification_f1", f"{cls_prefix}_f1", "max", args.f1_min, None),
        Goal("error_mae", f"{err_prefix}_mae", "min", args.mae_min, args.mae_max),
        Goal("error_rmse", f"{err_prefix}_rmse", "min", args.rmse_min, args.rmse_max),
        Goal("recommendation_pehe", "val_pehe", "min", args.pehe_min, args.pehe_max),
        Goal("recommendation_ate_error", "val_ate_error", "min", args.ate_min, args.ate_max),
    ]


def _build_all_metrics_goals(args: argparse.Namespace) -> list[Goal]:
    mse_min = args.mse_min
    mse_max = args.mse_max
    if mse_min is None and args.rmse_min is not None:
        mse_min = float(args.rmse_min) ** 2
    if mse_max is None and args.rmse_max is not None:
        mse_max = float(args.rmse_max) ** 2

    return [
        Goal("factual_auroc", "val_factual_auroc", "max", args.auroc_min, None),
        Goal("factual_auprc", "val_factual_auprc", "max", args.auprc_min, args.auprc_max),
        Goal("factual_f1", "val_factual_f1", "max", args.f1_min, None),
        Goal("factual_mae", "val_factual_mae", "min", args.mae_min, args.mae_max),
        Goal("factual_mse", "val_factual_mse", "min", mse_min, mse_max),
        Goal("factual_rmse", "val_factual_rmse", "min", args.rmse_min, args.rmse_max),

        Goal("sepsis_auroc", "val_sepsis_auroc", "max", args.auroc_min, None),
        Goal("sepsis_auprc", "val_sepsis_auprc", "max", args.auprc_min, args.auprc_max),
        Goal("sepsis_f1", "val_sepsis_f1", "max", args.f1_min, None),
        Goal("sepsis_mse", "val_sepsis_mse", "min", mse_min, mse_max),

        Goal("patient_sepsis_auroc", "val_patient_sepsis_auroc", "max", args.auroc_min, None),
        Goal("patient_sepsis_auprc", "val_patient_sepsis_auprc", "max", args.auprc_min, args.auprc_max),
        Goal("patient_sepsis_f1", "val_patient_sepsis_f1", "max", args.f1_min, None),

        Goal("sepsis_mae", "val_sepsis_mae", "min", args.mae_min, args.mae_max),
        Goal("sepsis_rmse", "val_sepsis_rmse", "min", args.rmse_min, args.rmse_max),
        Goal("patient_sepsis_mae", "val_patient_sepsis_mae", "min", args.mae_min, args.mae_max),
        Goal("patient_sepsis_mse", "val_patient_sepsis_mse", "min", mse_min, mse_max),
        Goal("patient_sepsis_rmse", "val_patient_sepsis_rmse", "min", args.rmse_min, args.rmse_max),

        Goal("recommendation_pehe", "val_pehe", "min", args.pehe_min, args.pehe_max),
        Goal("recommendation_ate_error", "val_ate_error", "min", args.ate_min, args.ate_max),
        Goal("recommendation_policy_regret", "val_policy_regret", "min", None, args.policy_regret_max),
    ]


def _build_goals(args: argparse.Namespace) -> list[Goal]:
    if args.goals_yaml is not None:
        return _load_goals_from_yaml(args.goals_yaml)
    if args.goal_profile == "all":
        return _build_all_metrics_goals(args)
    return _build_default_goals(args)


def _trial_checkpoint_path(study_dir: Path, trial_number: int) -> Path:
    return study_dir / f"trial_{trial_number:04d}" / "model" / "best_model.pt"


def _params_to_patch(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    patch: dict[str, dict[str, Any]] = {"model": {}, "train": {}}
    ff_multiplier: int | None = None

    for k, v in params.items():
        if "." not in k:
            continue
        root, leaf = k.split(".", 1)
        if root not in {"model", "train"}:
            continue
        if k == "model.ff_multiplier":
            ff_multiplier = int(v)
            continue
        patch.setdefault(root, {})[leaf] = v

    if ff_multiplier is not None and "hidden_size" in patch.get("model", {}):
        hidden = int(patch["model"]["hidden_size"])
        patch["model"]["ff_dim"] = int(hidden * ff_multiplier)

    return {k: v for k, v in patch.items() if v}


def _apply_patch_to_config(base_cfg: dict[str, Any], patch: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = json.loads(json.dumps(base_cfg))
    for section, values in patch.items():
        if section not in out or not isinstance(out[section], dict):
            out[section] = {}
        out[section].update(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick best trial checkpoint by explicit multi-goal rules (not Optuna objective)."
    )
    parser.add_argument("--study-dir", type=Path, default=None)
    parser.add_argument("--optuna-db", type=Path, default=None)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="Project out_dir used for tuning; used to infer study-dir.")

    parser.add_argument("--classification-level", type=str, default="patient", choices=["patient", "timestep"])
    parser.add_argument("--goal-profile", type=str, default="default", choices=["default", "all"])
    parser.add_argument("--goals-yaml", type=Path, default=None)

    parser.add_argument("--auroc-min", type=float, default=0.8)
    parser.add_argument("--auprc-min", type=float, default=0.4)
    parser.add_argument("--auprc-max", type=float, default=0.7)
    parser.add_argument("--f1-min", type=float, default=0.65)

    parser.add_argument("--mae-min", type=float, default=0.1)
    parser.add_argument("--mae-max", type=float, default=0.2)
    parser.add_argument("--rmse-min", type=float, default=0.15)
    parser.add_argument("--rmse-max", type=float, default=0.25)
    parser.add_argument("--mse-min", type=float, default=None)
    parser.add_argument("--mse-max", type=float, default=None)
    parser.add_argument("--pehe-min", type=float, default=0.1)
    parser.add_argument("--pehe-max", type=float, default=0.2)
    parser.add_argument("--ate-min", type=float, default=0.1)
    parser.add_argument("--ate-max", type=float, default=0.2)
    parser.add_argument("--policy-regret-max", type=float, default=0.05)
    parser.add_argument(
        "--strict-lower-bounds-for-min",
        action="store_true",
        help="When set, values below the min bound on min-metrics are treated as violations.",
    )

    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--out-best-json", type=Path, default=None)
    parser.add_argument("--out-ranked-json", type=Path, default=None)
    parser.add_argument("--out-patch-json", type=Path, default=None)
    parser.add_argument("--out-config-yaml", type=Path, default=None)
    parser.add_argument("--top-k-print", type=int, default=10)
    args = parser.parse_args()

    if args.study_dir is None:
        if args.out_dir is None:
            raise ValueError("Provide either --study-dir or --out-dir.")
        args.study_dir = _suggest_tune_root(args.out_dir, args.study_name)

    study_dir = args.study_dir.resolve()
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    db_path = args.optuna_db.resolve() if args.optuna_db else (study_dir / "optuna_study.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")

    goals = _build_goals(args)
    trials = _load_trials_from_optuna_db(db_path=db_path, study_name=args.study_name)
    complete_trials = [t for t in trials if t["state"] == "COMPLETE"]
    if not complete_trials:
        raise ValueError("No COMPLETE trials found yet.")

    ranked: list[dict[str, Any]] = []
    for t in complete_trials:
        attrs = t.get("user_attrs", {})
        metric_values = {g.key: _metric_value(attrs, g.key) for g in goals}

        goal_rows = []
        n_met = 0
        violation_total = 0.0
        utility_total = 0.0
        for g in goals:
            met, violation, utility = _score_goal(
                metric_values[g.key],
                g,
                strict_lower_bounds_for_min=bool(args.strict_lower_bounds_for_min),
            )
            goal_rows.append(
                {
                    "goal": g.name,
                    "metric_key": g.key,
                    "value": metric_values[g.key],
                    "met": met,
                    "violation": violation,
                    "utility": utility,
                    "min": g.min_value,
                    "max": g.max_value,
                }
            )
            if met:
                n_met += 1
            violation_total += float(violation)
            utility_total += float(utility)

        all_met = n_met == len(goals)
        trial_number = int(t["number"])
        ranked.append(
            {
                "trial_number": trial_number,
                "trial_id": int(t["trial_id"]),
                "state": t["state"],
                "objective_value": t.get("objective_value"),
                "all_goals_met": all_met,
                "n_goals_met": n_met,
                "n_goals_total": len(goals),
                "violation_total": violation_total,
                "utility_total": utility_total,
                "checkpoint_path": str(_trial_checkpoint_path(study_dir, trial_number)),
                "params": t.get("params", {}),
                "goal_details": goal_rows,
                "user_attrs": attrs,
            }
        )

    ranked.sort(
        key=lambda r: (
            -int(r["all_goals_met"]),
            -int(r["n_goals_met"]),
            float(r["violation_total"]),
            -float(r["utility_total"]),
            int(r["trial_number"]),
        )
    )
    best = ranked[0]
    patch = _params_to_patch(best.get("params", {}))

    out_best_json = args.out_best_json or (study_dir / "goal_selected_best.json")
    out_ranked_json = args.out_ranked_json or (study_dir / "goal_selected_ranked.json")
    out_patch_json = args.out_patch_json or (study_dir / "goal_selected_best_config_patch.json")

    out_best_json.parent.mkdir(parents=True, exist_ok=True)
    out_ranked_json.parent.mkdir(parents=True, exist_ok=True)
    out_patch_json.parent.mkdir(parents=True, exist_ok=True)

    out_best_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    out_ranked_json.write_text(
        json.dumps(
            {
                "goal_profile": args.goal_profile,
                "goals_yaml": (None if args.goals_yaml is None else str(args.goals_yaml)),
                "goals": [
                    {
                        "name": g.name,
                        "key": g.key,
                        "direction": g.direction,
                        "min": g.min_value,
                        "max": g.max_value,
                        "weight": g.weight,
                    }
                    for g in goals
                ],
                "ranked": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_patch_json.write_text(json.dumps(patch, indent=2), encoding="utf-8")

    if args.base_config is not None:
        base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
        merged = _apply_patch_to_config(base_cfg, patch)
        out_cfg = args.out_config_yaml or (study_dir / "goal_selected_config.yaml")
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        print(f"wrote_out_config={out_cfg}")

    print(f"study_dir={study_dir}")
    print(f"optuna_db={db_path}")
    print(f"goal_profile={args.goal_profile}")
    print(f"trials_complete={len(complete_trials)}")
    print(f"selected_trial={best['trial_number']}")
    print(f"all_goals_met={best['all_goals_met']}")
    print(f"n_goals_met={best['n_goals_met']}/{best['n_goals_total']}")
    print(f"violation_total={best['violation_total']:.6f}")
    print(f"checkpoint={best['checkpoint_path']}")
    print(f"wrote_best={out_best_json}")
    print(f"wrote_ranked={out_ranked_json}")
    print(f"wrote_patch={out_patch_json}")

    print("\nTop trials:")
    for row in ranked[: max(1, int(args.top_k_print))]:
        print(
            f"trial={row['trial_number']} "
            f"goals={row['n_goals_met']}/{row['n_goals_total']} "
            f"all_met={row['all_goals_met']} "
            f"violation={row['violation_total']:.4f} "
            f"utility={row['utility_total']:.4f} "
            f"objective={row['objective_value']}"
        )


if __name__ == "__main__":
    main()
