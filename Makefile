# Makefile for the OCF reproducibility pipeline.
#
# Usage:
#   make data        # Download + preprocess BRFSS 2024
#   make audit       # 5-fold CV training + per-method audit
#   make bootstrap   # 1000 stratified bootstrap replicates
#   make sensitivity # c-sweep over [0.5, 1.5]
#   make benchmark   # Compare with Fairlearn, AIF360, Aequitas
#   make verify-mlr  # Empirical MLR (A2') verification
#   make all         # Full pipeline (1+ hour)
#   make test        # pytest
#   make synth       # Quick smoke-test on synthetic data
#   make clean       # Remove auto-generated CSV / data files

PY := python3
DATA_DIR := ./data
RESULTS_DIR := ./results
MODEL := XGBoost
N_BOOTSTRAP := 1000

DATA_FILE := $(DATA_DIR)/brfss_2024_clean.parquet
SYNTH_FILE := $(DATA_DIR)/brfss_2024_SYNTHETIC.parquet

.PHONY: all data audit bootstrap sensitivity benchmark verify-mlr \
        synth test clean help

# Default target: print help
help:
	@echo "OCF reproducibility pipeline. Targets:"
	@echo "  make data        - download + preprocess BRFSS 2024"
	@echo "  make audit       - 5-fold CV training and audit (script 03)"
	@echo "  make bootstrap   - $(N_BOOTSTRAP) bootstrap replicates (script 04)"
	@echo "  make sensitivity - c-sweep (script 05)"
	@echo "  make benchmark   - compare with Fairlearn/AIF360/Aequitas (script 06)"
	@echo "  make verify-mlr  - MLR verification (script 07)"
	@echo "  make all         - full pipeline (>= 1 hour)"
	@echo "  make synth       - quick smoke test (synthetic data, < 1 min)"
	@echo "  make test        - run unit tests"
	@echo "  make clean       - remove generated results/data files"

# --- Pipeline steps ---

$(DATA_FILE):
	$(PY) scripts/01_download_brfss.py --data-dir $(DATA_DIR)
	$(PY) scripts/02_preprocess_brfss.py --data-dir $(DATA_DIR) --out $(DATA_FILE)

data: $(DATA_FILE)

audit: data
	$(PY) scripts/03_train_and_audit.py --data $(DATA_FILE) --out-dir $(RESULTS_DIR) --n-folds 5

bootstrap: data
	$(PY) scripts/04_bootstrap_audit.py \
	    --data $(DATA_FILE) \
	    --out-dir $(RESULTS_DIR) \
	    --model $(MODEL) \
	    --n-bootstrap $(N_BOOTSTRAP)

sensitivity: data
	$(PY) scripts/05_sensitivity_c.py \
	    --data $(DATA_FILE) \
	    --out-dir $(RESULTS_DIR) \
	    --model $(MODEL) \
	    --c-min 0.5 --c-max 1.5 --n-c 21

benchmark: data
	$(PY) scripts/06_benchmark_libraries.py \
	    --data $(DATA_FILE) \
	    --out-dir $(RESULTS_DIR) \
	    --model $(MODEL)

verify-mlr: data
	$(PY) scripts/07_verify_mlr.py \
	    --data $(DATA_FILE) \
	    --out-dir $(RESULTS_DIR)

all: data audit bootstrap sensitivity benchmark verify-mlr

# --- Validation runs ---

validate-brfss-2023:
	$(PY) scripts/validate_brfss_2023.py --data-dir $(DATA_DIR)

validate-nhanes:
	$(PY) scripts/validate_nhanes.py --data-dir $(DATA_DIR) --cycle 17-18

validate-meps:
	$(PY) scripts/validate_meps.py --data-dir $(DATA_DIR) --year 2022

# --- Synthetic-data smoke test (no CDC download required) ---

$(SYNTH_FILE):
	$(PY) scripts/00_synth_brfss.py --n 100000 --out $(SYNTH_FILE)

synth: $(SYNTH_FILE)
	$(PY) scripts/03_train_and_audit.py --data $(SYNTH_FILE) --out-dir $(RESULTS_DIR)/synth --n-folds 3
	$(PY) scripts/04_bootstrap_audit.py --data $(SYNTH_FILE) --out-dir $(RESULTS_DIR)/synth --model LogisticRegression --n-bootstrap 100 --weight-col none

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf $(RESULTS_DIR)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +

clean-data: clean
	rm -rf $(DATA_DIR)/*.parquet $(DATA_DIR)/*.XPT $(DATA_DIR)/*.zip
