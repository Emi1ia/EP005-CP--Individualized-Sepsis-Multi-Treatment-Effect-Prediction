#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def count_psv(folder: Path) -> int:
    return len(list(folder.glob("p*.psv")))


def check_challenge(root: Path) -> None:
    a_dir = root / "challenge-2019" / "training" / "training_setA"
    b_dir = root / "challenge-2019" / "training" / "training_setB"

    print("[challenge-2019]")
    if not a_dir.exists() or not b_dir.exists():
        print("  missing training_setA or training_setB")
        return

    a = count_psv(a_dir)
    b = count_psv(b_dir)
    print(f"  training_setA: {a} files")
    print(f"  training_setB: {b} files")
    print("  expected from challenge page: A=20336, B=20000")
    if a == 20336 and b == 20000:
        print("  status: complete")
    else:
        print("  status: incomplete or different version/mirror; run sync downloader")


def check_mimic(root: Path) -> None:
    hosp_dir = root / "mimiciv" / "3.1" / "hosp"
    icu_dir = root / "mimiciv" / "3.1" / "icu"
    print("[mimiciv-3.1]")

    if not hosp_dir.exists():
        print("  hosp module: missing")
    else:
        files = list(hosp_dir.glob("*.csv.gz"))
        print(f"  hosp module: {len(files)} csv.gz files")

    if not icu_dir.exists():
        print("  icu module: missing")
    else:
        files = list(icu_dir.glob("*.csv.gz"))
        print(f"  icu module: {len(files)} csv.gz files")

    required_hosp = [
        "admissions.csv.gz",
        "patients.csv.gz",
        "labevents.csv.gz",
        "microbiologyevents.csv.gz",
        "prescriptions.csv.gz",
        "pharmacy.csv.gz",
        "emar.csv.gz",
    ]
    required_icu = [
        "icustays.csv.gz",
        "inputevents.csv.gz",
        "chartevents.csv.gz",
        "outputevents.csv.gz",
        "procedureevents.csv.gz",
        "d_items.csv.gz",
    ]

    if hosp_dir.exists():
        missing = [f for f in required_hosp if not (hosp_dir / f).exists()]
        print(f"  hosp required missing: {missing if missing else 'none'}")
    if icu_dir.exists():
        missing = [f for f in required_icu if not (icu_dir / f).exists()]
        print(f"  icu required missing: {missing if missing else 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="c:/Users/emili/sepsis_project/data")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    print(f"data_root={root}")
    check_challenge(root)
    check_mimic(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
