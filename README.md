# ocf — Outcome-Calibrated Fairness for Binary Classification

[![Tests](https://img.shields.io/badge/tests-123%2F123-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Reference implementation of the algorithms, metrics, and experiments from:

> **Outcome-Calibrated Fairness for HIV Risk Prediction in High-Disparity Populations**
> [Author Names], 2026. Submitted to [JAMIA / Journal of Biomedical Informatics].

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

For multi-method comparison:

```python
out = evaluate_all(scores, y, a, seed=0)
for row in out["rows"]:
    print(f"{row['method']:<14} acc={row['accuracy']:.4f} "
          f"OCF_viol={row['OCF_violation']:.4f}")
# Unmitigated   acc=0.937 OCF_viol=0.054
# DP            acc=0.844 OCF_viol=0.283
# EO            acc=0.929 OCF_viol=0.051
# Calibration   acc=0.943 OCF_viol=0.015
# OCF           acc=0.941 OCF_viol=0.000
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

## What's new in v1.0.0

1. **Bootstrap c-bug fix.** Previous `04_bootstrap_audit.py` computed
   the OCF multiplier *c* using an unweighted mean of per-group base
   rates instead of the size-weighted mean, yielding `c ≈ 0.857`
   instead of the correct `c = 1.0`. The new bootstrap script uses
   `ocf.fit_ocf` directly. Headline numbers shift: Δ(OCF − DP) on
   high-burden TPR increases from ~12.9pp to ~20pp; Δ(OCF − Unmit)
   moves from −1.6pp to **+6.7pp positive**. See `CHANGELOG.md`.

2. **Proper Hardt-Price-Srebro EO.** The previous EO heuristic
   ("threshold matching median TPR/FPR") is replaced by the full
   LP-based HPS algorithm with randomized thresholds and exact equal-
   odds in expectation. Run-time is < 1 s for BRFSS-scale data.

3. **Standardized OCF violation metric.** Definition 3.6 introduces a
   single $L^\infty$ form normalized by max base rate. The legacy
   ratio-spread metric is retained as
   `ocf_violation_ratio_spread` for diagnostic continuity.

4. **Strengthened theoretical assumption.** A2 (within-group AUROC >
   0.5) retained as the audit condition; A2′ (MLR within group) added
   as the proof condition for Theorems 1, 3, 4. The rigorous proof of
   Theorem 3 is provided in `docs/SECTION_3_REVISIONS.md`.

5. **Benchmark integration.** `scripts/06_benchmark_libraries.py`
   compares OCF end-to-end against:
   - Fairlearn `ThresholdOptimizer` (DP + EO)
   - Fairlearn `ExponentiatedGradient` (DP)
   - AIF360 `Reweighing` + LR
   - AIF360 `EqOddsPostprocessing`
   - Aequitas audit on our predictions

6. **Sensitivity analysis.** New `ocf.sensitivity` module + script for
   sweeping the OCF screening multiplier *c* across $[0.5, 1.5]$,
   producing the appendix figure.

7. **Empirical MLR verification.** `scripts/07_verify_mlr.py` confirms
   A2′ holds for all 4 ML classifiers (LogReg, RF, MLP, XGBoost) on
   BRFSS 2024.

---

## Citation

If you use this code, please cite:

```bibtex
@article{ocf2026,
  author  = {[Authors]},
  title   = {Outcome-Calibrated Fairness for HIV Risk Prediction
             in High-Disparity Populations},
  journal = {[JAMIA / JBI / TBD]},
  year    = {2026},
  doi     = {[TBD]}
}
```

---

## License

MIT. See `LICENSE`.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example and the
expected vs. observed behavior.
# of
