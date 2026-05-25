"""Tests for ``ocf.metrics``."""
from __future__ import annotations

import numpy as np
import pytest

from ocf.metrics import (
    base_rate,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    positive_predictive_value,
    demographic_parity_difference,
    equalized_odds_difference,
    ocf_violation,
    ocf_violation_ratio_spread,
    calibration_disparity,
    _optimal_c,
)


# =============================================================================
# Group-wise rates
# =============================================================================

class TestGroupRates:
    def test_base_rate_simple(self):
        y = np.array([1, 1, 0, 0, 1])
        a = np.array(["A", "A", "B", "B", "A"])
        assert base_rate(y, a, "A") == pytest.approx(1.0)
        assert base_rate(y, a, "B") == pytest.approx(0.0)

    def test_base_rate_missing_group(self):
        assert np.isnan(base_rate(np.array([1, 0]), np.array(["A", "B"]), "C"))

    def test_selection_rate(self):
        y_hat = np.array([1, 0, 1, 0])
        a = np.array(["A", "A", "B", "B"])
        assert selection_rate(y_hat, a, "A") == 0.5
        assert selection_rate(y_hat, a, "B") == 0.5

    def test_tpr_and_fpr(self):
        # Perfect predictor inside group
        y = np.array([1, 1, 0, 0])
        y_hat = np.array([1, 1, 0, 0])
        a = np.array(["A", "A", "A", "A"])
        assert true_positive_rate(y_hat, y, a, "A") == 1.0
        assert false_positive_rate(y_hat, y, a, "A") == 0.0

    def test_ppv(self):
        # 3 positives predicted, 2 of which are correct
        y = np.array([1, 1, 0, 0])
        y_hat = np.array([1, 1, 1, 0])
        a = np.array(["A"] * 4)
        assert positive_predictive_value(y_hat, y, a, "A") == pytest.approx(2 / 3)


# =============================================================================
# Standard fairness metrics
# =============================================================================

class TestFairnessMetrics:
    def test_dpd_equal_rates(self):
        y_hat = np.array([1, 0, 1, 0, 1, 0])
        a = np.array(["A", "A", "B", "B", "C", "C"])
        assert demographic_parity_difference(y_hat, a) == pytest.approx(0.0)

    def test_dpd_full_disparity(self):
        # A always selected, B never selected.
        y_hat = np.array([1, 1, 0, 0])
        a = np.array(["A", "A", "B", "B"])
        assert demographic_parity_difference(y_hat, a) == pytest.approx(1.0)

    def test_eod_perfect_classifier(self):
        y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        a = np.array(["A"] * 4 + ["B"] * 4)
        y_hat = y.copy()
        assert equalized_odds_difference(y_hat, y, a) == pytest.approx(0.0)


# =============================================================================
# OCF violation — primary metric
# =============================================================================

class TestOCFViolation:
    def test_perfect_ocf(self):
        # Construct y_hat such that selection rate matches base rate
        rng = np.random.default_rng(0)
        n = 20000
        a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
        y = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y[mask] = rng.binomial(1, p, size=mask.sum())
        # y_hat: independent Bernoulli with prob = pi_a within each group
        y_hat = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y_hat[mask] = rng.binomial(1, p, size=mask.sum())
        viol = ocf_violation(y_hat, y, a)
        # Should be very small (only finite-sample noise)
        assert viol < 0.05

    def test_dp_high_violation(self):
        # DP fits all groups to same selection rate — OCF violation
        # should be large under base-rate disparity
        rng = np.random.default_rng(0)
        n = 20000
        a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
        y = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y[mask] = rng.binomial(1, p, size=mask.sum())
        # DP-like: every group has selection rate = 0.34 (mean(y))
        y_hat = np.zeros(n, dtype=int)
        bar = float(np.mean(y))
        for g in [0, 1, 2]:
            mask = a == g
            k = int(round(bar * mask.sum()))
            idx = rng.permutation(np.where(mask)[0])[:k]
            y_hat[idx] = 1
        viol = ocf_violation(y_hat, y, a)
        # Should be substantial — DP doesn't satisfy OCF
        assert viol > 0.15

    def test_normalization_bound(self):
        # OCF violation in [0, 1] when normalized
        rng = np.random.default_rng(1)
        n = 5000
        a = rng.choice(["X", "Y"], size=n)
        y = rng.binomial(1, 0.3, size=n)
        y_hat = rng.binomial(1, 0.3, size=n)
        v = ocf_violation(y_hat, y, a)
        assert 0 <= v <= 1

    def test_consistency_with_explicit_c(self):
        # If we know the data-generating c, OCF violation under that c
        # should match the auto-optimized value (or be larger, never smaller).
        rng = np.random.default_rng(2)
        n = 10000
        a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
        y = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y[mask] = rng.binomial(1, p, size=mask.sum())
        y_hat = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y_hat[mask] = rng.binomial(1, 0.5 * p, size=mask.sum())  # c=0.5
        v_auto = ocf_violation(y_hat, y, a)
        v_fixed = ocf_violation(y_hat, y, a, c=0.5)
        # Auto-optimized c should give smaller or equal violation
        assert v_auto <= v_fixed + 1e-6


class TestOptimalC:
    def test_optimal_c_exact_recovery(self):
        # rho = c * pi exactly → optimal c is c itself
        pi = np.array([0.2, 0.4, 0.6])
        c_true = 0.85
        rho = c_true * pi
        c_hat = _optimal_c(rho, pi)
        assert c_hat == pytest.approx(c_true, abs=1e-6)

    def test_optimal_c_noisy(self):
        # Add small perturbation, optimal c should be near c_true
        rng = np.random.default_rng(0)
        pi = np.array([0.2, 0.4, 0.6])
        c_true = 1.0
        rho = c_true * pi + rng.normal(0, 0.01, size=3)
        c_hat = _optimal_c(rho, pi)
        assert c_hat == pytest.approx(c_true, abs=0.05)


# =============================================================================
# Ratio-spread (legacy)
# =============================================================================

class TestRatioSpread:
    def test_zero_when_rho_proportional_to_pi(self):
        rng = np.random.default_rng(0)
        n = 10000
        a = rng.choice([0, 1, 2], size=n)
        pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
        y = np.zeros(n, dtype=int)
        y_hat = np.zeros(n, dtype=int)
        c = 0.7
        for g, p in pi_true.items():
            mask = a == g
            y[mask] = rng.binomial(1, p, size=mask.sum())
            y_hat[mask] = rng.binomial(1, c * p, size=mask.sum())
        v = ocf_violation_ratio_spread(y_hat, y, a)
        # Allow finite-sample noise; ratio-spread is more sensitive than L_inf
        assert v < 0.10


# =============================================================================
# Calibration disparity
# =============================================================================

class TestCalibrationDisparity:
    def test_zero_for_identical_distributions(self):
        rng = np.random.default_rng(0)
        n = 20000
        scores = rng.uniform(0, 1, size=n)
        y = rng.binomial(1, scores)
        a = rng.choice(["A", "B"], size=n)
        # If P(Y|score, A=A) ≈ P(Y|score, A=B), disparity should be small
        d = calibration_disparity(scores, y, a, n_bins=10)
        assert d < 0.15
