#!/usr/bin/env python3
"""
Preprocess BRFSS 2024 raw XPT data into an analysis-ready DataFrame.

Outputs a parquet file with:
  - Target variable:  hiv_tested_ever  (binary; from HIVTST7)
  - Protected attrs:  race_eth, msm_status (if SOMALE module present)
  - Demographics:     age_group, sex, education, income, marital
  - Geography:        state, metro_status, region (Census)
  - Healthcare:       has_insurance, regular_provider, cost_barrier
  - Behavioral:       binge_drink, smoking, exercise, alcohol
  - Survey weights:   _LLCPWT (if present; preserved for weighted analysis)

Usage:
    python 02_preprocess_brfss.py --data-dir ./data \\
                                   --out ./data/brfss_2024_clean.parquet

Reference BRFSS 2024 codebook:
    https://www.cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CORE_VARS = {
    "HIVTST7": "hiv_tested_ever_raw",
    "_IMPRACE": "race_eth_raw",
    "SEXVAR": "sex_raw",
    "SOMALE": "somale_raw",
    "SOFEMALE": "sofemale_raw",
    "TRNSGNDR": "trans_raw",
    "_AGEG5YR": "age_group_raw",
    "_AGE_G": "age_g_raw",
    "EDUCA": "education_raw",
    "INCOME3": "income_raw",
    "MARITAL": "marital_raw",
    "_STATE": "state_raw",
    "_METSTAT": "metro_raw",
    "_HLTHPL1": "insurance_raw",
    "PERSDOC3": "regular_provider_raw",
    "MEDCOST1": "cost_barrier_raw",
    "_RFBING6": "binge_drink_raw",
    "_RFSMOK3": "smoking_raw",
    "_RFDRHV8": "heavy_drink_raw",
    "_PA150R4": "physical_active_raw",
    "GENHLTH": "gen_health_raw",
    "_RFHLTH": "fair_poor_health_raw",
    "DIABETE4": "diabetes_raw",
    "_LLCPWT": "sample_weight_raw",
}


def clean_binary_yes_no(raw, yes_code=1, no_code=2):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == yes_code] = 1.0
    out[s == no_code] = 0.0
    return out


def clean_race_eth(raw):
    s = pd.to_numeric(raw, errors="coerce")
    mapping = {1: "White_NH", 2: "Black_NH", 3: "Asian_NH",
               4: "AIAN_NH", 5: "Hispanic", 6: "Other_NH"}
    return s.map(mapping).astype("category")


def clean_age_group(raw):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[s == 1] = "18-24"
    out[s.isin([2, 3])] = "25-34"
    out[s.isin([4, 5, 6, 7])] = "35-49"
    out[s.isin([8, 9, 10, 11, 12, 13])] = "50+"
    return pd.Categorical(out, categories=["18-24", "25-34", "35-49", "50+"],
                          ordered=True)


def clean_sex(raw):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def clean_msm_status(sex_raw, somale_raw):
    sex = pd.to_numeric(sex_raw, errors="coerce")
    so = pd.to_numeric(somale_raw, errors="coerce")
    out = pd.Series(np.nan, index=sex.index, dtype=float)
    male = sex == 1
    out[male & so.isin([1, 3])] = 1.0
    out[male & (so == 2)] = 0.0
    return out


def clean_education(raw):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[s.isin([1, 2, 3])] = "lt_HS"
    out[s == 4] = "HS_grad"
    out[s == 5] = "some_college"
    out[s == 6] = "college_grad"
    return pd.Categorical(out,
        categories=["lt_HS", "HS_grad", "some_college", "college_grad"],
        ordered=True)


def clean_income(raw):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[s.isin([1, 2, 3, 4])] = "lt_25k"
    out[s.isin([5, 6])] = "25_50k"
    out[s.isin([7, 8])] = "50_100k"
    out[s.isin([9, 10, 11])] = "ge_100k"
    return pd.Categorical(out,
        categories=["lt_25k", "25_50k", "50_100k", "ge_100k"], ordered=True)


def clean_marital(raw):
    s = pd.to_numeric(raw, errors="coerce")
    mapping = {1: "Married", 2: "Divorced", 3: "Widowed",
               4: "Separated", 5: "Never_married", 6: "Unmarried_couple"}
    return s.map(mapping).astype("category")


def clean_calc_binary_flag(raw):
    """Calc vars (_RFBING6, _RFSMOK3, etc.): 1=No, 2=Yes."""
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == 1] = 0.0
    out[s == 2] = 1.0
    return out


def clean_gen_health(raw):
    s = pd.to_numeric(raw, errors="coerce")
    return s.where(s.between(1, 5), np.nan).astype(float)


def clean_metro(raw):
    s = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def clean_state_region(raw):
    """Group FIPS state codes into US Census regions."""
    s = pd.to_numeric(raw, errors="coerce")
    south_fips = {1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54}
    northeast = {9, 23, 25, 33, 34, 36, 42, 44, 50}
    midwest = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
    west = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[s.isin(south_fips)] = "South"
    out[s.isin(northeast)] = "Northeast"
    out[s.isin(midwest)] = "Midwest"
    out[s.isin(west)] = "West"
    return pd.Categorical(out, categories=["South", "Northeast", "Midwest", "West"])


def preprocess(xpt_path: Path) -> pd.DataFrame:
    print(f"Loading {xpt_path}...")
    raw = pd.read_sas(xpt_path, format="xport", encoding="latin-1")
    print(f"  raw shape: {raw.shape}")
    present = [v for v in CORE_VARS if v in raw.columns]
    missing = [v for v in CORE_VARS if v not in raw.columns]
    print(f"  vars found: {len(present)}/{len(CORE_VARS)}")
    if missing:
        print(f"  missing: {missing}")

    df = pd.DataFrame(index=raw.index)
    target_var = None
    for cand in ("HIVTST7", "HIVTST6"):
        if cand in raw.columns:
            target_var = cand
            break
    if target_var is None:
        raise KeyError("No HIVTST7/HIVTST6 found - cannot proceed.")
    df["hiv_tested_ever"] = clean_binary_yes_no(raw[target_var])

    if "_IMPRACE" not in raw.columns:
        raise KeyError("_IMPRACE missing.")
    df["race_eth"] = clean_race_eth(raw["_IMPRACE"])

    sex_var = "SEXVAR" if "SEXVAR" in raw.columns else "_SEX"
    if sex_var in raw.columns:
        df["male"] = clean_sex(raw[sex_var])
    else:
        df["male"] = np.nan
    if "SOMALE" in raw.columns:
        df["msm"] = clean_msm_status(raw[sex_var], raw["SOMALE"])
    else:
        df["msm"] = np.nan
        print("  WARNING: SOMALE missing — MSM analysis unavailable.")

    if "_AGEG5YR" in raw.columns:
        df["age_group"] = clean_age_group(raw["_AGEG5YR"])
    if "EDUCA" in raw.columns:
        df["education"] = clean_education(raw["EDUCA"])
    if "INCOME3" in raw.columns:
        df["income"] = clean_income(raw["INCOME3"])
    elif "INCOME2" in raw.columns:
        df["income"] = clean_income(raw["INCOME2"])
    if "MARITAL" in raw.columns:
        df["marital"] = clean_marital(raw["MARITAL"])
    if "_METSTAT" in raw.columns:
        df["metro"] = clean_metro(raw["_METSTAT"])
    if "_STATE" in raw.columns:
        df["region"] = clean_state_region(raw["_STATE"])
    if "_HLTHPL1" in raw.columns:
        df["has_insurance"] = clean_binary_yes_no(raw["_HLTHPL1"])
    if "PERSDOC3" in raw.columns:
        s = pd.to_numeric(raw["PERSDOC3"], errors="coerce")
        df["regular_provider"] = s.isin([1, 2]).astype(float)
        df.loc[~s.isin([1, 2, 3]), "regular_provider"] = np.nan
    if "MEDCOST1" in raw.columns:
        df["cost_barrier"] = clean_binary_yes_no(raw["MEDCOST1"])
    if "_RFBING6" in raw.columns:
        df["binge_drink"] = clean_calc_binary_flag(raw["_RFBING6"])
    if "_RFSMOK3" in raw.columns:
        df["current_smoker"] = clean_calc_binary_flag(raw["_RFSMOK3"])
    if "_RFDRHV8" in raw.columns:
        df["heavy_drink"] = clean_calc_binary_flag(raw["_RFDRHV8"])
    if "_PA150R4" in raw.columns:
        s = pd.to_numeric(raw["_PA150R4"], errors="coerce")
        df["meets_aerobic"] = (s == 1).astype(float)
        df.loc[~s.isin([1, 2, 3, 9]), "meets_aerobic"] = np.nan
    if "GENHLTH" in raw.columns:
        df["gen_health"] = clean_gen_health(raw["GENHLTH"])
    if "DIABETE4" in raw.columns:
        df["diabetes"] = clean_binary_yes_no(raw["DIABETE4"])
    if "_LLCPWT" in raw.columns:
        df["_LLCPWT"] = pd.to_numeric(raw["_LLCPWT"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["hiv_tested_ever", "race_eth"])
    print(f"  after dropping missing target/race: {len(df):,} "
          f"(dropped {n_before - len(df):,})")
    return df


def report_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Distribution summaries")
    print("=" * 70)
    print(f"\nOverall HIV testing rate: P(tested) = "
          f"{df['hiv_tested_ever'].mean():.3f}")
    print("\nHIV testing rate by race/ethnicity:")
    by_race = df.groupby("race_eth", observed=True)["hiv_tested_ever"]
    summary = by_race.agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "p_tested"})
    print(summary.round(4).to_string())
    p_max = float(summary["p_tested"].max())
    p_min = float(summary["p_tested"].min())
    print(f"\n  Testing-rate ratio (max/min): {p_max/p_min:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./data/brfss_2024_clean.parquet")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    xpt_path = data_dir / "LLCP2024.XPT"
    if not xpt_path.exists():
        print(f"ERROR: {xpt_path} not found. Run 01_download_brfss.py first.")
        return 1
    df = preprocess(xpt_path)
    report_distribution(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved {len(df):,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
