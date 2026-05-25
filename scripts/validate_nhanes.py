#!/usr/bin/env python3
"""
External validation #2: NHANES 2017-2018 (true HIV serostatus).

NHANES is the gold-standard validation because it contains true HIV
serology results (LBXHIVC), not just self-reported testing uptake.
This is Interpretation A from the paper's "what does pi mean?"
appendix — replicating OCF with TRUE disease burden.

NHANES file structure:
    https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
    - DEMO_J.XPT (2017-2018 demographics)
    - HIV_J.XPT (HIV laboratory results 2017-2018)
    - SXQ_J.XPT (sexual behavior questionnaire)

Limitations:
    - Small N (~5,500 with HIV serology)
    - HIV serology only for adults 18-49 in public release
    - Restricted-access data required for full file

Usage:
    python validate_nhanes.py --data-dir ./data --cycle 17-18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

NHANES_FILES = {
    "17-18": {
        "demo": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT",
            "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT",
        ],
        "hiv": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HIV_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HIV_J.XPT",
            "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HIV_J.XPT",
        ],
        "sxq": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SXQ_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SXQ_J.XPT",
        ],
        "alq": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALQ_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALQ_J.XPT",
        ],
        "smq": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.XPT",
        ],
        "hiq": [
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HIQ_J.xpt",
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HIQ_J.XPT",
        ],
    },
}


def _is_valid_xpt(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return False
    with open(path, "rb") as f:
        header = f.read(80)
    return b"HEADER RECORD" in header and b"LIBRARY" in header


def _download_with_ua(url: str, dest: Path) -> tuple[bool, str]:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        with urlopen(req, timeout=120) as resp:
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def download_nhanes(data_dir: Path, cycle: str = "17-18") -> dict:
    files = NHANES_FILES[cycle]
    cycle_dir = data_dir / f"nhanes_{cycle}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, url_candidates in files.items():
        fname = url_candidates[0].rsplit("/", 1)[-1]
        fname_upper = fname.rsplit(".", 1)[0] + ".XPT"
        dest = cycle_dir / fname_upper
        if _is_valid_xpt(dest):
            print(f"  ✓ valid: {dest.name}")
            paths[key] = dest
            continue
        if dest.exists():
            dest.unlink()
        success = False
        last_error = ""
        for url in url_candidates:
            print(f"  trying URL: {url}")
            ok, err = _download_with_ua(url, dest)
            if not ok:
                last_error = err
                if dest.exists():
                    dest.unlink()
                continue
            if not _is_valid_xpt(dest):
                last_error = f"Non-XPT content from {url}"
                dest.unlink()
                continue
            print(f"    ✓ saved {dest.stat().st_size/1e6:.1f} MB")
            success = True
            break
        if not success:
            raise IOError(
                f"All URLs failed for NHANES '{key}'. Last error: {last_error}\n"
                f"Manual download: visit "
                f"https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/"
                f"default.aspx?BeginYear=2017 and place {fname_upper} at "
                f"{dest}."
            )
        paths[key] = dest
    return paths


def load_and_merge(paths: dict) -> pd.DataFrame:
    dfs = {}
    for key, path in paths.items():
        print(f"  loading {path.name}...")
        df = pd.read_sas(path, format="xport", encoding="latin-1")
        df = df.rename(columns={c: c.upper() for c in df.columns})
        dfs[key] = df
    merged = dfs["demo"]
    for key in ["hiv", "sxq", "alq", "smq", "hiq"]:
        if key in dfs:
            merged = merged.merge(dfs[key], on="SEQN", how="left")
    return merged


def preprocess_nhanes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hiv_var = "LBXHIVC" if "LBXHIVC" in df.columns else "LBXHIV"
    if hiv_var not in df.columns:
        raise KeyError("No HIV serology variable found")
    s = pd.to_numeric(df[hiv_var], errors="coerce")
    out["hiv_positive"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))

    s = pd.to_numeric(df["RIDRETH3"], errors="coerce")
    race_map = {1: "Hispanic", 2: "Hispanic", 3: "White_NH",
                4: "Black_NH", 6: "Asian_NH", 7: "Other_NH"}
    out["race_eth"] = s.map(race_map).astype("category")

    s = pd.to_numeric(df["RIAGENDR"], errors="coerce")
    out["male"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))

    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    out["age_group"] = pd.cut(age, bins=[0, 24, 34, 49, 200],
                              labels=["18-24", "25-34", "35-49", "50+"])

    if "HIQ011" in df.columns:
        s = pd.to_numeric(df["HIQ011"], errors="coerce")
        out["has_insurance"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))
    if "SMQ020" in df.columns:
        s = pd.to_numeric(df["SMQ020"], errors="coerce")
        out["current_smoker"] = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))
    if "SXQ410" in df.columns:
        s = pd.to_numeric(df["SXQ410"], errors="coerce")
        male_partners = (s > 0).astype(float)
        out["msm"] = np.where(
            (out["male"] == 1) & (male_partners == 1), 1.0,
            np.where(out["male"] == 1, 0.0, np.nan),
        )
    if "WTMEC2YR" in df.columns:
        out["sample_weight"] = pd.to_numeric(df["WTMEC2YR"], errors="coerce")
    if "SDMVSTRA" in df.columns:
        out["strata"] = pd.to_numeric(df["SDMVSTRA"], errors="coerce")
    if "SDMVPSU" in df.columns:
        out["psu"] = pd.to_numeric(df["SDMVPSU"], errors="coerce")

    out = out.dropna(subset=["hiv_positive", "race_eth"]).reset_index(drop=True)
    return out


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"NHANES HIV PREVALENCE BY RACE/ETHNICITY (n={len(df):,})")
    print("=" * 60)
    print(f"Overall HIV prevalence: {df['hiv_positive'].mean():.4f}")
    by_race = df.groupby("race_eth", observed=True)["hiv_positive"]
    print(by_race.agg(["count", "mean"]).round(4).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--cycle", default="17-18", choices=["17-18"])
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = download_nhanes(data_dir, cycle=args.cycle)
    raw = load_and_merge(paths)
    df = preprocess_nhanes(raw)
    report(df)
    out_path = data_dir / f"nhanes_{args.cycle}_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
