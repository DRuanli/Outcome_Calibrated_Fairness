# ocf — Outcome-Calibrated Fairness for Binary Classification

[![Tests](https://img.shields.io/badge/tests-123%2F123-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Reference implementation of the algorithms, metrics, and experiments from:

> **Outcome-Calibrated Fairness for HIV Risk Prediction in High-Disparity Populations**
> Dang Nguyen Le, Minh-Anh Tran Nguyen, Yamin Thiri Wai, Duc-Nhan Tran, 2026. Submitted to Journal of Biomedical Informatics.

The package implements **Algorithms 1–4** of the paper:

| Algorithm | What it does | Module |
|---|---|---|
| 1 | OCF post-processing (this paper's contribution) | `ocf.postprocess.fit_ocf` |
| 2 | Demographic-parity post-processing (Hardt-Price-Srebro, threshold-rule) | `ocf.postprocess.fit_dp` |
| 3 | Equalized-odds post-processing (Hardt-Price-Srebro, full LP+randomized) | `ocf.eo_hardt.fit_eo_hardt` |
| 4 | Group-wise isotonic calibration | `ocf.postprocess.fit_calibration` |

Plus standard fairness metrics (DPD, EOD), the paper's **OCF violation**
metric in its corrected $L^\infty$ form (Definition 3.6), and utilities
for sensitivity analysis, MLR verification, and stratified bootstrap.

---

## Quick start

```python
import numpy as np
from ocf import fit_ocf, predict_ocf, ocf_violation, evaluate_all

# scores: predicted P(Y=1 | X) from any classifier on validation data
# a: protected attribute (race/ethnicity, etc.)
# y: binary labels

# Fit OCF post-processor (Algorithm 1)
result = fit_ocf(scores, a, y)
print(f"Screening multiplier c = {result.c:.4f}")      # 1.0 by default
print(f"Per-group thresholds: {result.thresholds}")

# Apply to test data
y_hat = predict_ocf(scores_test, a_test, result)

# Compute OCF violation (Definition 3.6, L_inf form)
viol = ocf_violation(y_hat, y_test, a_test)
print(f"OCF violation = {viol:.6f}")  # should be ~0 after fit
```
---

## Installation

```bash
# Core dependencies only (sufficient for using the API)
pip install -e .

# With parquet I/O (for running pipeline scripts)
pip install -e .[data]

# With XGBoost (one of the 4 ML baselines in the paper)
pip install -e .[xgboost]

# With third-party fairness libraries (for the benchmark script)
pip install -e .[benchmark]

# Everything (recommended for full reproducibility)
pip install -e .[all]
```

After install, run the unit tests:

```bash
pytest tests/ -v
# 123 passed in ~3s
```

---

## Reproducing the paper

The experimental pipeline has 7 scripts (in `scripts/`):

| Step | Script | Purpose | Runtime |
|---:|---|---|---|
| 0 | `00_synth_brfss.py` | Generate synthetic BRFSS data for testing | ~5 s |
| 1 | `01_download_brfss.py` | Download BRFSS 2024 raw XPT from CDC | ~30 s |
| 2 | `02_preprocess_brfss.py` | Clean and preprocess into analysis schema | ~2 min |
| 3 | `03_train_and_audit.py` | Cross-validation training + per-method audit | ~10 min |
| 4 | `04_bootstrap_audit.py` | Stratified bootstrap CIs (1000 replicates) | ~30 min |
| 5 | `05_sensitivity_c.py` | Sensitivity sweep of OCF multiplier *c* | ~5 min |
| 6 | `06_benchmark_libraries.py` | Compare vs Fairlearn / AIF360 / Aequitas | ~10 min |
| 7 | `07_verify_mlr.py` | Empirical MLR (A2′) verification | ~10 min |

Plus three external-data validators:

- `validate_brfss_2023.py` — cross-year robustness (BRFSS 2023)
- `validate_nhanes.py` — true HIV serostatus (NHANES 2017-2018)
- `validate_meps.py` — clinical utilization (MEPS 2022)

### Full reproduction recipe

```bash
# 1. Get the data
python scripts/01_download_brfss.py --data-dir ./data
python scripts/02_preprocess_brfss.py --data-dir ./data \
    --out ./data/brfss_2024_clean.parquet

# 2. Main experiment (Table 2 of the paper)
python scripts/03_train_and_audit.py \
    --data ./data/brfss_2024_clean.parquet \
    --n-folds 5

# 3. Bootstrap CIs (Table 3 / Figure 2)
python scripts/04_bootstrap_audit.py \
    --data ./data/brfss_2024_clean.parquet \
    --n-bootstrap 1000 \
    --model XGBoost

# 4. Sensitivity (Figure S1)
python scripts/05_sensitivity_c.py \
    --data ./data/brfss_2024_clean.parquet \
    --model XGBoost \
    --c-min 0.5 --c-max 1.5 --n-c 21

# 5. Third-party library benchmark (Appendix Table A4)
python scripts/06_benchmark_libraries.py \
    --data ./data/brfss_2024_clean.parquet \
    --model XGBoost

# 6. MLR verification (Section 3.7)
python scripts/07_verify_mlr.py \
    --data ./data/brfss_2024_clean.parquet
```

For convenience, the `Makefile` wraps all of these:

```bash
make data        # download + preprocess BRFSS 2024
make audit       # 5-fold CV training + audit (script 03)
make bootstrap   # 1000 bootstrap replicates (script 04)
make sensitivity # c-sweep (script 05)
make benchmark   # third-party comparison (script 06)
make verify-mlr  # MLR check (script 07)
make all         # full pipeline (≥1 hour)
make test        # run pytest
make synth       # quick smoke test on synthetic data
```

---

## Package structure

```
ocf/
├── ocf/                            # Core library
│   ├── __init__.py                 # Public API re-exports
│   ├── metrics.py                  # base/selection/TPR/FPR/PPV, DPD, EOD,
│   │                                 OCF violation (L_inf, ratio-spread)
│   ├── postprocess.py              # Algorithms 1, 2, 4 + unmitigated baseline
│   ├── eo_hardt.py                 # Algorithm 3 (HPS-LP, randomized)
│   ├── evaluation.py               # evaluate_all + stratified_bootstrap_audit
│   ├── sensitivity.py              # c-sweep helpers, bisection search
│   └── theory.py                   # MLR check, Bayes-optimal rates
│
├── scripts/                        # Pipeline scripts (see table above)
│   ├── 00_synth_brfss.py           ├── 04_bootstrap_audit.py
│   ├── 01_download_brfss.py        ├── 05_sensitivity_c.py
│   ├── 02_preprocess_brfss.py      ├── 06_benchmark_libraries.py
│   ├── 03_train_and_audit.py       ├── 07_verify_mlr.py
│   ├── validate_brfss_2023.py      ├── validate_nhanes.py
│   └── validate_meps.py
│
├── tests/                          # Unit tests (123, all passing)
│   ├── test_metrics.py             ├── test_eo_hardt.py
│   ├── test_postprocess.py         ├── test_theorems.py
│   ├── test_sensitivity.py         └── test_theory.py
│
├── docs/
│   ├── SECTION_3_REVISIONS.md      # Theoretical patch for Section 3
│   └── REPRODUCIBILITY.md          # Script-to-table mapping
│
├── results/                        # Auto-generated CSVs
├── pyproject.toml                  # Package metadata + dependencies
├── Makefile                        # Reproducibility targets
├── CHANGELOG.md                    # Version history (bug-fix notes)
└── README.md                       # This file
```

---

## Citation

If you use this code, please cite:

```bibtex
@unpublished{le2026ocf,
  author = {Le, Nguyen and Tran, M. A. Nguyen and Wai, Yamin Thiri
            and Tran, Duc-Nhan and Duong, Huu-Phuoc},
  title  = {Outcome-Calibrated Fairness: A Base-Rate-Aware Post-Processing
            Criterion for {HIV} Risk Prediction under Demographic Disparity},
  note   = {Preprint submitted to Journal of Biomedical Informatics},
  year   = {2026}
}
```

---

## License

MIT. See `LICENSE`.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example and the
expected vs. observed behavior.
# OFC
