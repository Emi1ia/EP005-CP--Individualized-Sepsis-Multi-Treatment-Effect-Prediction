# Dataset Review

## Scope reviewed

- PhysioNet 2019 sepsis challenge:
  - `training_setA` (20,139 patients, 782,749 hourly rows)
  - `training_setB` (19,909 patients, 758,733 hourly rows)
- Local MIMIC demo sample:
  - `mimic_demo/icu/icustays.csv.gz` has 140 ICU stays (100 subjects)

## Key findings

1. **No explicit treatment administration columns in PhysioNet 2019 files.**
   - Missing expected treatment fields for antibiotics, fluids, and vasopressors.
   - This blocks direct real-world individualized treatment-effect learning.

2. **Strong label imbalance (sepsis-positive is rare).**
   - Set A: septic patients = 1,774/20,139 (8.81%)
   - Set B: septic patients = 1,134/19,909 (5.70%)
   - Row-level positive fraction:
     - Set A: 2.17%
     - Set B: 1.41%
     - Combined: ~1.80%

3. **Very high missingness for many labs and gases.**
   - Combined high-missing examples:
     - `Bilirubin_direct`: 99.81% missing
     - `Fibrinogen`: 99.34%
     - `TroponinI`: 99.05%
     - `Lactate`: 97.33%
     - `PTT`: 97.06%
     - `SaO2`: 96.56%
     - `EtCO2`: 96.28%
   - Low-missing signals are mostly vitals/demographics (`HR`, `MAP`, `O2Sat`, `SBP`, `Age`, `Gender`, `ICULOS`).

4. **Set A vs Set B distribution differences (domain shift).**
   - Sepsis prevalence differs materially (8.81% vs 5.70% by patient).
   - Missingness profiles differ (for example very sparse chemistry/gas variables in both sets, but not identically sparse).

5. **Data integrity is generally good.**
   - `ICULOS` monotonic issues: none detected.
   - Label reversions (1 back to 0): none detected.
   - Plausibility checks: mostly clean, with a small number of out-of-range observations.

6. **MIMIC demo data is too small for robust treatment-effect training.**
   - Only 140 ICU stays.
   - Useful for code tests and extraction prototypes, not for final model claims.

## Is current dataset "bad"?

- For **sepsis onset/risk prediction**, the PhysioNet data is usable.
- For **proposal-grade multi-treatment causal effect estimation**, current data is **insufficient** without additional treatment/outcome tables.

## What is needed next

1. **Real treatment data** with timestamps and doses/categories:
   - antibiotics administration/ordering,
   - IV fluids bolus/rate/volume,
   - vasopressor type and intensity.
2. **Outcome targets aligned to proposal**:
   - 24-hour shock progression,
   - 48-hour organ dysfunction change (for example SOFA deltas),
   - in-hospital mortality.
3. **Sepsis anchor/recognition timestamp** to align trajectories consistently.
4. **Larger MIMIC cohort (full access, not demo subset)** for treatment effect estimation and external validation.

