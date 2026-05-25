#!/usr/bin/env python3
"""
External validation #1: BRFSS 2023 (cross-year robustness).

Identical pipeline to BRFSS 2024 but for the 2023 release. Confirms that
OCF results are not artifacts of a single year and that the
strengthened MLR assumption (A2') holds robustly.

BRFSS 2023 vs 2024:
  - 2023 has the SOGI module (sexual orientation/gender identity) in
    some states. 2024 was modified under executive orders.
  - HIV testing variable: HIVTST7 (same name as 2024).
  - Race: _IMPRACE (same).

Usage:
    python validate_brfss_2023.py --data-dir ./data
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BRFSS_2023_URL = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"


def download_brfss_2023(data_dir: Path) -> Path:
    zip_path = data_dir / "LLCP2023XPT.zip"
    xpt_path = data_dir / "LLCP2023.XPT"
    if xpt_path.exists():
        return xpt_path
    print(f"Downloading BRFSS 2023 from {BRFSS_2023_URL}...")
    urlretrieve(BRFSS_2023_URL, zip_path)
    print(f"Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            clean = name.strip()
            if clean.upper().endswith(".XPT"):
                with zf.open(name) as src, open(data_dir / clean, "wb") as dst:
                    dst.write(src.read())
                print(f"  extracted: {clean}")
    return xpt_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    xpt_path = download_brfss_2023(data_dir)

    print(f"\nReading XPT ({xpt_path})...")
    here = Path(__file__).resolve().parent
    preprocess_script = here / "02_preprocess_brfss.py"
    if not preprocess_script.exists():
        for candidate in (
            here.parent / "scripts" / "02_preprocess_brfss.py",
            Path.cwd() / "scripts" / "02_preprocess_brfss.py",
        ):
            if candidate.exists():
                preprocess_script = candidate
                break
        else:
            raise FileNotFoundError("Cannot find 02_preprocess_brfss.py")

    spec = importlib.util.spec_from_file_location("pre_mod", str(preprocess_script))
    pre_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_mod)

    df = pre_mod.preprocess(xpt_path)
    pre_mod.report_distribution(df)
    out_path = data_dir / "brfss_2023_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")
    print(f"\nNext: re-run 03_train_and_audit.py with --data {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
