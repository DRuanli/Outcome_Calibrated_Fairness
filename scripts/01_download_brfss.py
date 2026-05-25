#!/usr/bin/env python3
"""
Download BRFSS 2024 raw data from CDC.

Downloads the SAS Transport (XPT) format file, which is the official
public BRFSS 2024 dataset (N = 457,670 records, 345 variables).

Usage:
    python 01_download_brfss.py [--data-dir DATA_DIR]

After successful run, data will be in:
    DATA_DIR/LLCP2024.XPT      (raw XPT, ~700 MB unzipped)
    DATA_DIR/LLCP2024XPT.zip   (downloaded zip, 64 MB)

If you have trouble downloading from the CDC, alternative sources:
    - Kaggle: https://www.kaggle.com/datasets/rudritarahman/cdc-brfss-survey-data-2024
    - OSF mirror (older years only): https://osf.io/6rxf4
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

BRFSS_2024_URL = "https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
BRFSS_2024_ZIP_NAME = "LLCP2024XPT.zip"
BRFSS_2024_XPT_NAME = "LLCP2024.XPT"
EXPECTED_ZIP_SIZE_MB = 60


def download_with_progress(url: str, dest: Path) -> None:
    def reporthook(blocknum, blocksize, totalsize):
        downloaded = blocknum * blocksize
        if totalsize > 0:
            pct = min(100.0, downloaded * 100.0 / totalsize)
            mb_down = downloaded / 1e6
            mb_total = totalsize / 1e6
            sys.stdout.write(
                f"\r  Downloaded {mb_down:6.1f} / {mb_total:6.1f} MB ({pct:5.1f}%)"
            )
            sys.stdout.flush()
    urlretrieve(url, str(dest), reporthook)
    sys.stdout.write("\n")


def check_url(url: str) -> bool:
    try:
        with urlopen(url, timeout=15) as resp:
            return resp.status == 200
    except (URLError, Exception):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / BRFSS_2024_ZIP_NAME
    xpt_path = data_dir / BRFSS_2024_XPT_NAME

    if xpt_path.exists() and not args.force:
        size_mb = xpt_path.stat().st_size / 1e6
        print(f"✓ XPT already exists: {xpt_path} ({size_mb:.1f} MB)")
        return 0

    print(f"Target URL: {BRFSS_2024_URL}")
    print(f"Destination: {zip_path}")

    if not zip_path.exists() or args.force:
        if not check_url(BRFSS_2024_URL):
            print(f"\nERROR: Cannot reach CDC URL {BRFSS_2024_URL}")
            print(f"Workaround: download manually to {zip_path}, then re-run with --force.")
            return 1
        print("Downloading BRFSS 2024 (~60 MB)...")
        try:
            download_with_progress(BRFSS_2024_URL, zip_path)
        except Exception as exc:
            print(f"\nDownload failed: {exc}")
            return 1

    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"  Contents: {names}")
        for name in names:
            clean_name = name.strip()
            if not clean_name.upper().endswith((".XPT", ".SAS")):
                continue
            target = data_dir / clean_name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  Extracted: {name!r} -> {target.name}")

    candidates = [p for p in data_dir.iterdir()
                  if p.is_file() and p.name.strip().upper().endswith(".XPT")]
    if not candidates:
        print("ERROR: No .XPT file found.")
        return 1
    actual_xpt = candidates[0]
    if actual_xpt.name != BRFSS_2024_XPT_NAME:
        actual_xpt.rename(data_dir / BRFSS_2024_XPT_NAME)
    size_mb = xpt_path.stat().st_size / 1e6
    print(f"\n✓ Done. {xpt_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
