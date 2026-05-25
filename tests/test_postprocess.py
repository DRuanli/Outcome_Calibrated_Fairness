"""Tests for ``ocf.postprocess`` — Algorithms 1, 2, 4, unmitigated."""
from __future__ import annotations

import numpy as np
import pytest

from ocf.postprocess import (
    fit_ocf,
    predict_ocf,
    fit_dp,
    predict_dp,
    fit_calibration,
    predict_calibration,
    fit_unmitigated,
    predict_unmitigated,
    apply_per_group_thresholds,
    _quantile_threshold,
)
from ocf.metrics import (
    selection_rate,
    base_rate,
    demographic_parity_difference,
    ocf_violation,
)


# =============================================================================
# Synthetic-data fixtures
# =============================================================================

@pytest.fixture
def simple_disparity():
    """3-group dataset with base rates pi = [0.2, 0.4, 0.6]."""
    rng = np.random.default_rng(0)
    n = 30000
    a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
    y = np.zeros(n, dtype=int)
    for g, p in pi_true.items():
        mask = a == g
        y[mask] = rng.binomial(1, p, size=mask.sum())
    scores = 1.5 * y + 0.3 * (a - 1) + rng.normal(0, 0.5, n)
    return scores, y, a, pi_true


# =============================================================================
# Quantile threshold utility
# =============================================================================

class TestQuantileThreshold:
    def test_target_rate_zero(self):
        assert _quantile_threshold(np.linspace(0, 1, 100), 0.0) == float("inf")

    def test_target_rate_one(self):
        assert _quantile_threshold(np.linspace(0, 1, 100), 1.0) == float("-inf")

    def test_target_rate_half(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 10000)
        t = _quantile_threshold(scores, 0.5)
        rate = float(np.mean(scores >= t))
        assert abs(rate - 0.5) < 0.02


# =============================================================================
# Algorithm 1: OCF
# =============================================================================

class TestOCF:
    def test_default_c_is_one(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_ocf(scores, a, y)
        assert res.c == pytest.approx(1.0, abs=1e-6)

    def test_target_aggregate_matches_mean_y_by_default(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_ocf(scores, a, y)
        assert res.target_aggregate_rate == pytest.approx(float(np.mean(y)),
                                                         abs=1e-6)

    def test_pi_hat_matches_truth(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_ocf(scores, a, y)
        for g, pi in pi_true.items():
            assert res.pi_hat[g] == pytest.approx(pi, abs=0.01)

    def test_selection_rate_matches_target(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_ocf(scores, a, y)
        y_hat = predict_ocf(scores, a, res)
        for g in pi_true:
            rho_g = selection_rate(y_hat, a, g)
            assert rho_g == pytest.approx(res.target_rho[g], abs=0.01)

    def test_ocf_violation_near_zero_after_fit(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_ocf(scores, a, y)
        y_hat = predict_ocf(scores, a, res)
        viol = ocf_violation(y_hat, y, a)
        # Only finite-sample noise from quantile lookup
        assert viol < 0.01

    def test_explicit_c_param(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        for c in [0.5, 0.75, 1.0, 1.25, 1.5]:
            res = fit_ocf(scores, a, y, c=c)
            assert res.c == c
            y_hat = predict_ocf(scores, a, res)
            for g in pi_true:
                rho_g = selection_rate(y_hat, a, g)
                target = c * pi_true[g]
                assert rho_g == pytest.approx(target, abs=0.02)

    def test_sample_weight_uniform_matches_unweighted(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res_u = fit_ocf(scores, a, y)
        weights = np.ones_like(y, dtype=float)
        res_w = fit_ocf(scores, a, y, sample_weight=weights)
        assert res_u.c == pytest.approx(res_w.c, abs=1e-10)
        for g in pi_true:
            assert res_u.pi_hat[g] == pytest.approx(res_w.pi_hat[g], abs=1e-10)

    def test_sample_weight_doubles_one_group(self, simple_disparity):
        # Doubling the weight of group 0 (lowest pi) should pull
        # the aggregate target rate down and require c > 1 to
        # match a higher target.
        scores, y, a, pi_true = simple_disparity
        weights = np.where(a == 0, 2.0, 1.0)
        res = fit_ocf(scores, a, y, sample_weight=weights)
        # By the c = mean_w(y) / mean_w(pi) formula, since both numerator
        # and denominator shift in lock-step (mean_w(y) ≡ mean_w(pi) by
        # tower property), c should still be 1.0.
        assert res.c == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Algorithm 2: DP
# =============================================================================

class TestDP:
    def test_equal_selection_rates(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        res = fit_dp(scores, a, y=y)
        y_hat = predict_dp(scores, a, res)
        rates = [selection_rate(y_hat, a, g) for g in pi_true]
        assert max(rates) - min(rates) < 0.01

    def test_aggregate_matches_target(self, simple_disparity):
        scores, y, a, pi_true = simple_disparity
        target = float(np.mean(y))
        res = fit_dp(scores, a, target_rate=target)
        y_hat = predict_dp(scores, a, res)
        assert float(np.mean(y_hat)) == pytest.approx(target, abs=0.01)

    def test_explicit_target(self, simple_disparity):
        scores, y, a, _ = simple_disparity
        for target in [0.1, 0.3, 0.5, 0.7]:
            res = fit_dp(scores, a, target_rate=target)
            y_hat = predict_dp(scores, a, res)
            assert float(np.mean(y_hat)) == pytest.approx(target, abs=0.01)


# =============================================================================
# Algorithm 4: Calibration
# =============================================================================

class TestCalibration:
    def test_runs_and_returns_predictions(self, simple_disparity):
        scores, y, a, _ = simple_disparity
        # scale scores to [0, 1] for isotonic
        from scipy.special import expit
        s_prob = expit(scores)
        res = fit_calibration(s_prob, a, y, threshold=0.5)
        y_hat = predict_calibration(s_prob, a, res)
        assert y_hat.shape == y.shape
        assert set(np.unique(y_hat)) <= {0, 1}

    def test_small_group_passes_through(self):
        # Group with < min_group_size: calibrator should be None
        scores = np.linspace(0, 1, 100)
        y = np.array([0, 1] * 50)
        a = np.array(["big"] * 90 + ["tiny"] * 10)
        res = fit_calibration(scores, a, y, min_group_size=50)
        assert res.calibrators["tiny"] is None
        assert res.calibrators["big"] is not None


# =============================================================================
# Unmitigated baseline
# =============================================================================

class TestUnmitigated:
    def test_aggregate_matches_target(self, simple_disparity):
        scores, y, a, _ = simple_disparity
        res = fit_unmitigated(scores, y)
        y_hat = predict_unmitigated(scores, res)
        assert float(np.mean(y_hat)) == pytest.approx(float(np.mean(y)), abs=0.01)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_apply_thresholds_missing_group(self):
        # Group present in `a` but missing from thresholds: predicted 0
        scores = np.array([0.6, 0.7, 0.8])
        a = np.array(["A", "B", "C"])
        thresholds = {"A": 0.5, "B": 0.5}  # C missing
        y_hat = apply_per_group_thresholds(scores, a, thresholds)
        assert y_hat[0] == 1 and y_hat[1] == 1
        assert y_hat[2] == 0  # default

    def test_all_zero_base_rate_raises(self):
        scores = np.array([0.5, 0.6, 0.7])
        a = np.array(["A", "A", "A"])
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError):
            fit_ocf(scores, a, y)
