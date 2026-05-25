#!/usr/bin/env python3
"""
External validation #3: MEPS (Medical Expenditure Panel Survey).

MEPS is run by AHRQ and provides healthcare utilization data,
including HIV testing in select cycles. This is Interpretation B/C
from the paper's "what does pi mean?" appendix — clinical utilization
in a non-behavioral survey.

NOTE: MEPS 2022 (HC-243) public file does NOT include a dedicated HIV
testing variable in the publicly-available portion. The script will
fall back to using a related proxy (blood-test uptake) if no HIV
variable is found, and warn the user that BRFSS or NHANES is preferred
for HIV-specific OCF validation.

Usage:
    python validate_meps.py --year 2022 --data-dir ./data
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

MEPS_URLS = {
    2022: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h243/h243v9.zip",
    2021: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h233/h233v9.zip",
}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _download_with_ua(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=180) as resp:
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)


def download_meps(data_dir: Path, year: int = 2022) -> Path:
    if year not in MEPS_URLS:
        raise NotImplementedError(f"Year {year} not configured.")
    url = MEPS_URLS[year]
    zip_path = data_dir / f"meps_{year}.zip"
    cands = (list(data_dir.glob("h*.XPT")) + list(data_dir.glob("h*.xpt"))
             + list(data_dir.glob("h*.sas7bdat")))
    if cands:
        for c in cands:
            if c.stat().st_size > 1_000_000:
                print(f"  ✓ Found: {c.name}")
                return c
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print(f"Downloading MEPS {year} from {url}...")
        _download_with_ua(url, zip_path)

    print(f"Extracting {zip_path}...")
    extracted = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            print(f"  in zip: {name}")
            if name.lower().endswith((".xpt", ".sas7bdat", ".ssp")):
                dest = data_dir / Path(name).name
                with zf.open(name) as src, open(dest, "wb") as out:
                    out.write(src.read())
                if extracted is None:
                    extracted = dest
    if extracted is None:
        raise IOError(f"No data file in {zip_path}")
    return extracted


def preprocess_meps(data_path: Path) -> pd.DataFrame:
    print(f"Loading {data_path}...")
    suffix = data_path.suffix.lower()
    if suffix in (".xpt",):
        raw = pd.read_sas(data_path, format="xport", encoding="latin-1")
    elif suffix == ".sas7bdat":
        raw = pd.read_sas(data_path, format="sas7bdat", encoding="latin-1")
    else:
        try:
            raw = pd.read_sas(data_path, format="sas7bdat", encoding="latin-1")
        except Exception:
            raw = pd.read_sas(data_path, format="xport", encoding="latin-1")
    raw = raw.rename(columns={c: c.upper() for c in raw.columns})
    print(f"  shape: {raw.shape}")

    hiv_candidates = [c for c in raw.columns if "HIV" in c.upper() or "AIDS" in c.upper()]
    print(f"  HIV-related columns: {hiv_candidates}")

    df = pd.DataFrame(index=raw.index)
    hiv_var = None
    for cand in ["HIVTEST22", "HIVTEST_M22", "HIVTEST"]:
        if cand in raw.columns:
            hiv_var = cand
            break
    if hiv_var is None and hiv_candidates:
        hiv_var = hiv_candidates[0]

    if hiv_var:
        s = pd.to_numeric(raw[hiv_var], errors="coerce")
        df["hiv_tested_ever"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))
    else:
        print("\n*** WARNING: No HIV testing variable found in MEPS file. ***")
        print("    Falling back to a blood-test uptake proxy.")
        for proxy in ["ADBLDS42", "BLOODT22", "ADCHLC42"]:
            if proxy in raw.columns:
                s = pd.to_numeric(raw[proxy], errors="coerce")
                df["hiv_tested_ever"] = np.where(
                    s == 1, 1.0, np.where(s == 2, 0.0, np.nan)
                )
                print(f"    Using proxy: {proxy}")
                break

    race_var = None
    for cand in ["RACETHX", "RACETHX_M", "RACETHX22", "RACETHX_M22"]:
        if cand in raw.columns:
            race_var = cand
            break
    if race_var is None:
        cands = [c for c in raw.columns if "RACETH" in c]
        if cands:
            race_var = cands[0]
    s = pd.to_numeric(raw[race_var], errors="coerce")
    race_map = {1: "Hispanic", 2: "White_NH", 3: "Black_NH",
                4: "Asian_NH", 5: "Other_NH"}
    df["race_eth"] = s.map(race_map).astype("category")

    if "SEX" in raw.columns:
        s = pd.to_numeric(raw["SEX"], errors="coerce")
        df["male"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))

    for age_cand in ["AGE22X", "AGE21X", "AGELAST"]:
        if age_cand in raw.columns:
            age = pd.to_numeric(raw[age_cand], errors="coerce")
            df["age_group"] = pd.cut(age, bins=[0, 24, 34, 49, 200],
                                    labels=["18-24", "25-34", "35-49", "50+"])
            break

    for ins_cand in ["INSCOV22", "INSCOV21", "INSCOV"]:
        if ins_cand in raw.columns:
            s = pd.to_numeric(raw[ins_cand], errors="coerce")
            df["has_insurance"] = np.where(s.isin([1, 2]), 1.0,
                                          np.where(s == 3, 0.0, np.nan))
            break

    if "EDUCYR" in raw.columns:
        edu = pd.to_numeric(raw["EDUCYR"], errors="coerce")
        df["education"] = pd.cut(edu, bins=[-1, 11, 12, 15, 30],
                                labels=["lt_HS", "HS_grad",
                                        "some_college", "college_grad"])
    for w_cand in ["PERWT22F", "PERWT21F", "PERWT"]:
        if w_cand in raw.columns:
            df["sample_weight"] = pd.to_numeric(raw[w_cand], errors="coerce")
            break

    df = df.dropna(subset=["race_eth"]).reset_index(drop=True)
    if "hiv_tested_ever" in df.columns:
        df = df.dropna(subset=["hiv_tested_ever"]).reset_index(drop=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    xpt_path = download_meps(data_dir, year=args.year)
    df = preprocess_meps(xpt_path)
    print("\n" + "=" * 60)
    print(f"MEPS {args.year} REPORT (n={len(df):,})")
    print("=" * 60)
    if "hiv_tested_ever" in df.columns and df["hiv_tested_ever"].notna().any():
        print(f"Overall target rate: {df['hiv_tested_ever'].mean():.4f}")
        by_race = df.groupby("race_eth", observed=True)["hiv_tested_ever"]
        print(by_race.agg(["count", "mean"]).round(4).to_string())
    else:
        print("\n*** MEPS does not have an HIV variable in this file. ***")
        print("    Use BRFSS or NHANES for HIV-specific OCF validation.")
        print(df.groupby("race_eth", observed=True).size().to_string())
    out_path = data_dir / f"meps_{args.year}_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
