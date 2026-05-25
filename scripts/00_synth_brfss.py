"""
Generate a synthetic dataset that mimics BRFSS 2024 structure for testing
the pipeline before real CDC data is available.

Mimics: variable names, value ranges, missing rates, racial composition
roughly matching US adult population (~63% White NH, 12% Black NH,
6% Asian NH, 1% AIAN NH, 17% Hispanic, 1% Other).

Key target: HIV testing rate by race that resembles published BRFSS
patterns - Black NH ~50% ever tested, Hispanic ~38%, White NH ~32%,
Asian NH ~25% (this is the disparity that motivates OCF analysis).

Output: ``data/brfss_2024_SYNTHETIC.parquet``

Usage:
    python 00_synth_brfss.py --n 400000 --out data/brfss_2024_SYNTHETIC.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_brfss(n: int = 400_000, seed: int = 20260523) -> pd.DataFrame:
    """Generate n rows of BRFSS-like data with realistic disparities."""
    rng = np.random.default_rng(seed)

    race_probs = {
        "White_NH": 0.62, "Black_NH": 0.12, "Hispanic": 0.17,
        "Asian_NH": 0.06, "AIAN_NH": 0.01, "Other_NH": 0.02,
    }
    races = list(race_probs.keys())
    probs = np.array(list(race_probs.values()))
    probs = probs / probs.sum()
    race = rng.choice(races, size=n, p=probs)

    male = (rng.uniform(size=n) < 0.49).astype(float)
    age = rng.choice(["18-24", "25-34", "35-49", "50+"], size=n,
                     p=[0.10, 0.18, 0.27, 0.45])
    region = rng.choice(["South", "Northeast", "Midwest", "West"], size=n,
                        p=[0.38, 0.17, 0.21, 0.24])
    edu = rng.choice(["lt_HS", "HS_grad", "some_college", "college_grad"],
                     size=n, p=[0.10, 0.27, 0.30, 0.33])

    income = []
    for e in edu:
        if e == "lt_HS":
            p = [0.50, 0.30, 0.15, 0.05]
        elif e == "HS_grad":
            p = [0.30, 0.35, 0.25, 0.10]
        elif e == "some_college":
            p = [0.20, 0.30, 0.30, 0.20]
        else:
            p = [0.10, 0.20, 0.30, 0.40]
        income.append(rng.choice(["lt_25k", "25_50k", "50_100k", "ge_100k"], p=p))
    income = np.array(income)

    marital = rng.choice(
        ["Married", "Divorced", "Widowed", "Separated",
         "Never_married", "Unmarried_couple"],
        size=n, p=[0.48, 0.13, 0.06, 0.02, 0.27, 0.04])

    metro_p = np.where(region == "South", 0.78, 0.86)
    metro = (rng.uniform(size=n) < metro_p).astype(float)

    insurance_p = np.where(income == "lt_25k", 0.78, 0.92)
    insurance_p = np.where(race == "Hispanic", insurance_p - 0.10, insurance_p)
    has_insurance = (rng.uniform(size=n) < insurance_p).astype(float)
    regular_provider = (rng.uniform(size=n) <
                       (0.85 - 0.15 * (1 - has_insurance))).astype(float)
    cost_barrier = (rng.uniform(size=n) <
                   (0.10 + 0.15 * (1 - has_insurance))).astype(float)
    binge_drink = (rng.uniform(size=n) <
                  np.where(male == 1, 0.20, 0.12)).astype(float)
    current_smoker = (rng.uniform(size=n) < 0.16).astype(float)
    heavy_drink = (rng.uniform(size=n) < 0.06).astype(float)
    meets_aerobic = (rng.uniform(size=n) < 0.51).astype(float)
    gen_health = rng.choice([1, 2, 3, 4, 5], size=n,
                            p=[0.18, 0.32, 0.30, 0.15, 0.05]).astype(float)
    diabetes = (rng.uniform(size=n) < 0.11).astype(float)

    # Generate HIV testing target with realistic racial disparities.
    logit = -0.5 * np.ones(n)
    race_effect = {"White_NH": 0.0, "Black_NH": 0.95, "Hispanic": 0.50,
                   "Asian_NH": -0.40, "AIAN_NH": 0.30, "Other_NH": 0.10}
    logit += np.array([race_effect[r] for r in race])
    age_effect = {"18-24": -0.30, "25-34": 0.30, "35-49": 0.10, "50+": -0.40}
    logit += np.array([age_effect[age_v] for age_v in age])
    logit += -0.10 * male
    logit += np.where(region == "South", 0.20, 0.0)
    logit += 0.30 * binge_drink + 0.15 * current_smoker
    logit += 0.30 * has_insurance
    logit += -0.20 * cost_barrier
    p_tested = 1.0 / (1.0 + np.exp(-logit))
    hiv_tested_ever = (rng.uniform(size=n) < p_tested).astype(float)

    def add_missing(arr, rate=0.02):
        mask = rng.uniform(size=n) < rate
        if arr.dtype == float:
            arr = arr.copy()
            arr[mask] = np.nan
        else:
            arr = pd.Series(arr).astype(object)
            arr.loc[mask] = np.nan
            arr = arr.values
        return arr

    has_insurance = add_missing(has_insurance, 0.03)
    regular_provider = add_missing(regular_provider, 0.03)
    cost_barrier = add_missing(cost_barrier, 0.02)
    binge_drink = add_missing(binge_drink, 0.05)
    current_smoker = add_missing(current_smoker, 0.03)
    heavy_drink = add_missing(heavy_drink, 0.06)
    meets_aerobic = add_missing(meets_aerobic, 0.04)
    gen_health = add_missing(gen_health, 0.01)
    diabetes = add_missing(diabetes, 0.01)
    income = add_missing(income, 0.08)
    education = add_missing(edu, 0.02)
    marital = add_missing(marital, 0.02)

    df = pd.DataFrame({
        "hiv_tested_ever": hiv_tested_ever, "race_eth": race, "male": male,
        "age_group": age, "education": education, "income": income,
        "marital": marital, "region": region, "metro": metro,
        "has_insurance": has_insurance, "regular_provider": regular_provider,
        "cost_barrier": cost_barrier, "binge_drink": binge_drink,
        "current_smoker": current_smoker, "heavy_drink": heavy_drink,
        "meets_aerobic": meets_aerobic, "gen_health": gen_health,
        "diabetes": diabetes,
    })
    df["race_eth"] = pd.Categorical(df["race_eth"], categories=races)
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=["18-24", "25-34", "35-49", "50+"], ordered=True
    )
    df["region"] = pd.Categorical(
        df["region"], categories=["South", "Northeast", "Midwest", "West"]
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--out", default="./data/brfss_2024_SYNTHETIC.parquet")
    ap.add_argument("--seed", type=int, default=20260523)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_brfss(n=args.n, seed=args.seed)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
    print(f"\nDistribution preview:")
    print(df["race_eth"].value_counts(normalize=True).round(3))
    print(f"\nHIV testing rate by race:")
    print(df.groupby("race_eth", observed=True)
          ["hiv_tested_ever"].agg(["count", "mean"]).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
