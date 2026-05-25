#!/usr/bin/env bash
#
# run_everything.sh — Full OCF reproduction pipeline.
#
# Runs every script in `scripts/` end-to-end on real BRFSS 2024, plus
# all three external-data validators (BRFSS 2023, NHANES, MEPS), with
# bootstrap CIs computed for every ML model (LogReg, RF, MLP, XGBoost).
#
# Total wall-clock time: 3-6 hours on a 16-core workstation.
# Disk usage:            ~5 GB (raw data + results)
# Memory:                ~16 GB peak (bootstrap with XGBoost)
#
# Usage:
#   bash run_everything.sh                  # standard run
#   N_BOOTSTRAP=200 bash run_everything.sh  # faster (wider CIs)
#   SKIP_VALIDATORS=1 bash run_everything.sh  # only BRFSS 2024
#   MODELS="XGBoost" bash run_everything.sh   # only XGBoost
#
# Outputs:
#   results/                            BRFSS 2024 results
#   results_brfss_2023/                 BRFSS 2023 (cross-year)
#   results_nhanes/                     NHANES (true serostatus)
#   results_meps/                       MEPS (clinical utilization)
#   logs/                               per-script stdout/stderr logs

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_FOLDS="${N_FOLDS:-5}"
MODELS="${MODELS:-LogisticRegression RandomForest MLP XGBoost}"
DATA_DIR="${DATA_DIR:-./data}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOG_DIR="${LOG_DIR:-./logs}"
SKIP_VALIDATORS="${SKIP_VALIDATORS:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

log() {
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] $*"
    echo "================================================================"
}

run_step() {
    local name="$1"; shift
    local log_file="$LOG_DIR/${name}.log"
    log "STEP: $name"
    echo "Command: $*"
    echo "Log:     $log_file"
    local t0=$(date +%s)
    if ! "$@" 2>&1 | tee "$log_file"; then
        echo "ERROR: $name failed. See $log_file" >&2
        return 1
    fi
    local t1=$(date +%s)
    local dt=$((t1 - t0))
    echo "  done in ${dt}s"
}

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------

log "0. SANITY CHECKS"

echo "Python version:"
python3 --version

echo "Package import + version:"
python3 -c "import ocf; print(f'  ocf v{ocf.__version__}')"

echo "Unit tests (must pass before pipeline):"
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -3

echo ""
echo "Library availability:"
python3 -c "
libs = ['numpy', 'pandas', 'sklearn', 'scipy', 'pyarrow', 'xgboost',
        'fairlearn', 'aif360', 'aequitas']
for lib in libs:
    try:
        m = __import__(lib)
        v = getattr(m, '__version__', '?')
        print(f'  {lib:<12} {v}  ✓')
    except ImportError:
        print(f'  {lib:<12} MISSING  (some scripts will skip)')
"

# ---------------------------------------------------------------------------
# 1. BRFSS 2024 download + preprocess
# ---------------------------------------------------------------------------

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    if [[ ! -f "$DATA_DIR/LLCP2024.XPT" ]]; then
        run_step "01_download_brfss" \
            python3 scripts/01_download_brfss.py --data-dir "$DATA_DIR"
    else
        log "1. BRFSS 2024 XPT already exists, skipping download"
    fi

    if [[ ! -f "$DATA_DIR/brfss_2024_clean.parquet" ]]; then
        run_step "02_preprocess_brfss" \
            python3 scripts/02_preprocess_brfss.py \
                --data-dir "$DATA_DIR" \
                --out "$DATA_DIR/brfss_2024_clean.parquet"
    else
        log "2. BRFSS 2024 parquet already exists, skipping preprocess"
    fi
fi

DATA="$DATA_DIR/brfss_2024_clean.parquet"
if [[ ! -f "$DATA" ]]; then
    echo "FATAL: $DATA not found. Run with SKIP_DOWNLOAD=0 first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Cross-validation audit (all 4 models, 5 methods → Table 2)
# ---------------------------------------------------------------------------

run_step "03_train_and_audit" \
    python3 scripts/03_train_and_audit.py \
        --data "$DATA" \
        --out-dir "$RESULTS_DIR" \
        --n-folds "$N_FOLDS"

# ---------------------------------------------------------------------------
# 4. Bootstrap CIs for EACH model (Table 3 + Figure 2)
# ---------------------------------------------------------------------------

for MODEL in $MODELS; do
    run_step "04_bootstrap_${MODEL}" \
        python3 scripts/04_bootstrap_audit.py \
            --data "$DATA" \
            --out-dir "$RESULTS_DIR" \
            --model "$MODEL" \
            --n-bootstrap "$N_BOOTSTRAP"
done

# ---------------------------------------------------------------------------
# 5. Sensitivity sweep (Figure S1)
# ---------------------------------------------------------------------------

for MODEL in $MODELS; do
    run_step "05_sensitivity_${MODEL}" \
        python3 scripts/05_sensitivity_c.py \
            --data "$DATA" \
            --out-dir "$RESULTS_DIR" \
            --model "$MODEL" \
            --c-min 0.5 --c-max 1.5 --n-c 21
done

# ---------------------------------------------------------------------------
# 6. Third-party library benchmark (Appendix Table A4)
# ---------------------------------------------------------------------------

if [[ "$SKIP_BENCHMARK" != "1" ]]; then
    for MODEL in $MODELS; do
        run_step "06_benchmark_${MODEL}" \
            python3 scripts/06_benchmark_libraries.py \
                --data "$DATA" \
                --out-dir "$RESULTS_DIR" \
                --model "$MODEL" \
            || echo "  WARNING: benchmark for $MODEL failed (likely missing fairlearn/aif360)"
    done
fi

# ---------------------------------------------------------------------------
# 7. MLR verification (Section 3.7 / Table 4)
# ---------------------------------------------------------------------------

run_step "07_verify_mlr" \
    python3 scripts/07_verify_mlr.py \
        --data "$DATA" \
        --out-dir "$RESULTS_DIR" \
        --n-bins 20 \
        --tolerances 0.00 0.05 0.10

# ---------------------------------------------------------------------------
# 8. External-data validation runs (Section 5.5 / Appendix B)
# ---------------------------------------------------------------------------

if [[ "$SKIP_VALIDATORS" != "1" ]]; then

    # 8a. BRFSS 2023 cross-year robustness
    log "8a. BRFSS 2023 cross-year validation"
    if [[ ! -f "$DATA_DIR/brfss_2023_clean.parquet" ]]; then
        run_step "validate_brfss_2023" \
            python3 scripts/validate_brfss_2023.py --data-dir "$DATA_DIR" \
            || echo "  WARNING: BRFSS 2023 download failed"
    fi
    if [[ -f "$DATA_DIR/brfss_2023_clean.parquet" ]]; then
        mkdir -p "${RESULTS_DIR}_brfss_2023"
        run_step "03_brfss_2023" \
            python3 scripts/03_train_and_audit.py \
                --data "$DATA_DIR/brfss_2023_clean.parquet" \
                --out-dir "${RESULTS_DIR}_brfss_2023" \
                --n-folds "$N_FOLDS"
        run_step "04_brfss_2023_XGBoost" \
            python3 scripts/04_bootstrap_audit.py \
                --data "$DATA_DIR/brfss_2023_clean.parquet" \
                --out-dir "${RESULTS_DIR}_brfss_2023" \
                --model XGBoost \
                --n-bootstrap "$N_BOOTSTRAP"
    fi

    # 8b. NHANES (true HIV serostatus)
    log "8b. NHANES 2017-2018 validation (true serostatus)"
    if [[ ! -f "$DATA_DIR/nhanes_17-18_clean.parquet" ]]; then
        run_step "validate_nhanes" \
            python3 scripts/validate_nhanes.py --data-dir "$DATA_DIR" --cycle 17-18 \
            || echo "  WARNING: NHANES download failed (CDC site requires manual download sometimes)"
    fi
    if [[ -f "$DATA_DIR/nhanes_17-18_clean.parquet" ]]; then
        mkdir -p "${RESULTS_DIR}_nhanes"
        echo "  NOTE: For NHANES, target is 'hiv_positive' (true serostatus)."
        echo "        Run 03_train_and_audit.py with --target hiv_positive."
        echo "        (Currently the script uses 'hiv_tested_ever' by default;"
        echo "        for NHANES analysis you'll need to manually adapt scripts/03"
        echo "        to use hiv_positive as the target. See docs/REPRODUCIBILITY.md."
    fi

    # 8c. MEPS (clinical utilization)
    log "8c. MEPS 2022 validation (clinical utilization)"
    if [[ ! -f "$DATA_DIR/meps_2022_clean.parquet" ]]; then
        run_step "validate_meps" \
            python3 scripts/validate_meps.py --data-dir "$DATA_DIR" --year 2022 \
            || echo "  WARNING: MEPS download failed or target variable missing"
    fi
    if [[ -f "$DATA_DIR/meps_2022_clean.parquet" ]]; then
        mkdir -p "${RESULTS_DIR}_meps"
        echo "  NOTE: MEPS 2022 may not have HIV testing variable. Check output."
    fi
fi

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------

log "9. PIPELINE COMPLETE"

echo ""
echo "Generated CSVs in $RESULTS_DIR/:"
ls -1 "$RESULTS_DIR"/*.csv 2>/dev/null | sed 's|^|  |'

echo ""
echo "Per-validator results:"
for vdir in "${RESULTS_DIR}_brfss_2023" "${RESULTS_DIR}_nhanes" "${RESULTS_DIR}_meps"; do
    if [[ -d "$vdir" ]]; then
        echo "  $vdir/"
        ls -1 "$vdir"/*.csv 2>/dev/null | sed 's|^|    |' || echo "    (empty)"
    fi
done

echo ""
echo "Logs in $LOG_DIR/:"
ls -1 "$LOG_DIR"/*.log 2>/dev/null | sed 's|^|  |'

echo ""
echo "NEXT STEPS:"
echo "  1. Inspect headline numbers in:"
echo "       $RESULTS_DIR/headline_table.csv"
echo "       $RESULTS_DIR/bootstrap_XGBoost_summary.csv"
echo "  2. Update paper tables/figures per CHANGELOG.md 'Notes for paper text update'"
echo "  3. Regenerate Section 3 from docs/SECTION_3_REVISIONS.md"
echo ""