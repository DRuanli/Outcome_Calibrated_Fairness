# Reproducibility Guide

This document maps every table, figure, and headline number in the
paper to the script that produces it. All scripts are in `scripts/`
and assume the package is installed via `pip install -e .[all]` (see
`README.md` for details).

---

## Quick reference: paper artifact → script

| Paper artifact | Script | Output file | Runtime |
|---|---|---|---:|
| **Table 2** (CV audit, all 4 models × 5 methods) | `03_train_and_audit.py` | `results/all_metrics.csv`, `results/headline_table.csv` | ~10 min |
| **Table 3** (bootstrap 95% CI for XGBoost) | `04_bootstrap_audit.py --model XGBoost` | `results/bootstrap_XGBoost_summary.csv` | ~30 min |
| **Figure 2** (Δ-TPR forest plot across methods) | `04_bootstrap_audit.py` (raw CSV → plot in R/matplotlib) | `results/bootstrap_XGBoost_raw.csv` | (uses Table 3 output) |
| **Figure S1** (sensitivity to *c*) | `05_sensitivity_c.py` | `results/sensitivity_c_XGBoost.csv` | ~5 min |
| **Appendix Table A4** (vs Fairlearn/AIF360/Aequitas) | `06_benchmark_libraries.py` | `results/benchmark_XGBoost.csv` | ~10 min |
| **Section 3.7** (MLR robustness, all 4 models) | `07_verify_mlr.py` | `results/mlr_verification.csv` | ~10 min |
| **Section 5.5** (BRFSS 2023 robustness) | `validate_brfss_2023.py` + re-run script 03 | `results/brfss_2023_*.csv` | ~15 min |
| **Appendix B** (NHANES validation, true serostatus) | `validate_nhanes.py` + re-run script 03 | `data/nhanes_*_clean.parquet` | ~15 min |
| **Appendix B** (MEPS clinical utilization) | `validate_meps.py` + re-run script 03 | `data/meps_*_clean.parquet` | ~15 min |

---

## End-to-end recipe

```bash
# 1. Setup
git clone <repo-url> ocf && cd ocf
pip install -e .[all]
pytest tests/ -v        # expect 123/123 in ~3s

# 2. Get BRFSS 2024
make data
# → ./data/brfss_2024_clean.parquet  (≈ 450k rows, ≈ 50 MB)

# 3. Reproduce paper tables in order
make audit        # → Table 2
make bootstrap    # → Table 3, Figure 2 (data)
make sensitivity  # → Figure S1
make benchmark    # → Appendix Table A4
make verify-mlr   # → Section 3.7 robustness numbers

# 4. External validations (Appendix)
make validate-brfss-2023
make validate-nhanes
make validate-meps
```

Total wall-clock time: ~1.5 h on a 16-core workstation, dominated by
the bootstrap (1000 replicates × 4 methods × 4 ML models if you re-run
audit on every model; the default `--model XGBoost` runs only the
headline model).

---

## Per-section reproduction details

### Section 5.1 (Headline result): "OCF beats DP on Black NH TPR"

The headline numbers in the paper's abstract and Section 5.1 come from:

- **Point estimate** of `TPR_Black_NH` for OCF vs DP vs Unmitigated:
  `results/headline_table.csv` from `03_train_and_audit.py`.
- **95% CI** for the gap: `results/bootstrap_XGBoost_summary.csv` from
  `04_bootstrap_audit.py`.

To check: open `bootstrap_XGBoost_summary.csv` and inspect the row
where `method == "OCF"` and compare `TPR_Black_NH_pt` ± its CI vs the
same row for `DP`.

### Section 5.2 (All ML models): "OCF advantage is model-agnostic"

`03_train_and_audit.py` runs all 4 base learners (LogReg, RF, MLP,
XGBoost) automatically and produces a 4 × 5 metrics table per group.
Headline numbers come from `summary.csv` aggregated across folds.

To regenerate just one model's bootstrap, use `--model RandomForest` etc.

### Section 5.3 (Theorem verification across bootstrap)

`04_bootstrap_audit.py` ends with a "[theorem checks]" block printing
the percent of replicates in which each theorem holds. These percentages
appear in the paper as: "Theorem 1 holds in X% of replicates," etc.

If using the v1.0.0 fix, expect **100%** for Theorems 1 and 4 on BRFSS
2024 with XGBoost. (The previous buggy version reported ~87% for
Theorem 4; this was an artifact, not a theory failure.)

### Section 3.7 (Empirical MLR verification)

`07_verify_mlr.py` produces `mlr_verification.csv` with one row per
`(model, group, tolerance)` triple. The paper reports, for each base
learner, the fraction of race/ethnicity groups in which MLR holds
under three tolerances (`0.00`, `0.05`, `0.10`). Numbers go in
**Table 4** of the revised Section 3.7.

To generate the Table 4 row for, e.g., XGBoost at tolerance 0.05:

```python
import pandas as pd
df = pd.read_csv("results/mlr_verification.csv")
sub = df[(df["model"] == "XGBoost") & (df["tolerance"] == 0.05)]
print(f"{sub['mlr_holds'].mean()*100:.0f}% of groups pass MLR (XGBoost, tol=0.05)")
```

### Figure S1 (Sensitivity to *c*)

`05_sensitivity_c.py` writes `sensitivity_c_XGBoost.csv` with columns:
`c, accuracy, TPR_<group>, rho_<group>, FPR_<group>, PPV_<group>` for
each group, plus convenience shortcuts `TPR_high_burden` and
`rho_high_burden`. Plot `c` vs each metric column to reproduce S1.

Suggested matplotlib code:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/sensitivity_c_XGBoost.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
groups = ["White_NH", "Black_NH", "Hispanic", "Asian_NH"]
for g in groups:
    axes[0].plot(df["c"], df[f"TPR_{g}"], label=g)
    axes[1].plot(df["c"], df[f"rho_{g}"], label=g)
axes[2].plot(df["c"], df["accuracy"], color="black")
axes[2].plot(df["c"], df["OCF_violation"], color="red", label="OCF viol")
for ax, t in zip(axes, ["Group TPR vs c", "Group selection rate vs c",
                       "Accuracy / OCF violation vs c"]):
    ax.set_xlabel("c (screening multiplier)")
    ax.set_title(t)
    ax.legend()
plt.tight_layout()
plt.savefig("figure_s1_sensitivity.pdf")
```

### Appendix Table A4 (Library benchmark)

`06_benchmark_libraries.py` produces `benchmark_XGBoost.csv` with one
row per `(library, method)` pair. The paper's Appendix Table A4 is
this CSV pivoted to compare OCF (ours) vs Fairlearn DP/EO/ExpGrad vs
AIF360 Reweighing/EqOddsPostproc.

Suggested column order in paper: `Library`, `Method`, `Accuracy`,
`DPD`, `EOD`, `OCF_violation`, `TPR_high_burden`. The
script's final printout already shows this layout.

### Appendix B (External validations)

Each validator script downloads its respective dataset, preprocesses
to the same schema (`hiv_tested_ever` or `hiv_positive`, `race_eth`,
demographics), and saves a parquet file. To run the OCF audit on the
external dataset:

```bash
python scripts/validate_nhanes.py --data-dir ./data
python scripts/03_train_and_audit.py \
    --data ./data/nhanes_17-18_clean.parquet \
    --out-dir ./results/nhanes
```

The protected attribute is still `race_eth`, but the target becomes
`hiv_positive` for NHANES (true serostatus).

---

## Common pitfalls

### "Why are my OCF results different from the paper?"

Five likely causes:

1. **Different BRFSS year**: the paper uses 2024. Cross-year results
   (BRFSS 2023) are reported in Section 5.5 and are slightly
   different — this is the robustness check, not a bug.
2. **Sample weights**: pass `--weight-col _LLCPWT` to scripts
   `03` and `04` for weighted analysis. Without weights, the OCF
   gap is somewhat larger (BRFSS oversamples certain demographics).
3. **Random seed**: most scripts default to `--seed 42`. Reproducing
   any specific bootstrap replicate requires the same seed.
4. **Pre-v1.0.0 bug**: if your numbers match "Δ(OCF − Unmit) ≈ −1.6 pp",
   you have the old c-computation bug. Re-run with v1.0.0 of this
   package and the gap flips to +6.7 pp.
5. **XGBoost version**: XGBoost RNG behavior varies slightly between
   2.0.x and 2.1.x. Use the version pinned in `pyproject.toml`
   (`xgboost>=2.0`) for closest match.

### "I get `ImportError: pyarrow not found`"

```bash
pip install -e .[data]    # or pip install pyarrow
```

### "I get `ImportError: aif360.tensorflow_ops`"

AIF360 imports TensorFlow lazily and the benchmark script does not
use that path. Safe to ignore as long as `from aif360.algorithms.preprocessing import Reweighing` works.

### "Bootstrap is too slow"

Reduce `--n-bootstrap` (default 1000) to e.g. 200 for a faster check.
CIs will be wider but point estimates are unchanged.

### "Synthetic data results look weird"

`scripts/00_synth_brfss.py` generates data where AUROC ≈ 0.60 (the
score is fairly noisy) — this is enough to demonstrate the OCF/DP
pattern qualitatively but underestimates the gap. Real BRFSS gives
AUROC ≈ 0.79 with XGBoost.
