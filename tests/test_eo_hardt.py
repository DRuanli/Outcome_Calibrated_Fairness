"""Tests for ``ocf.eo_hardt`` — Hardt-Price-Srebro Equalized Odds."""
from __future__ import annotations

import numpy as np
import pytest

from ocf.eo_hardt import (
    fit_eo_hardt,
    predict_eo_hardt,
    _group_roc,
    _upper_hull,
)
from ocf.metrics import (
    true_positive_rate,
    false_positive_rate,
    equalized_odds_difference,
)


# =============================================================================
# Helper components
# =============================================================================

class TestROCBuilder:
    def test_perfect_classifier(self):
        # Scores perfectly separate positives from negatives
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        y = np.array([1, 1, 0, 0])
        roc = _group_roc(scores, y)
        # ROC should pass near (0, 1)
        assert any(np.allclose(p, [0.0, 1.0], atol=0.01) for p in roc)

    def test_random_classifier_roc_is_diagonal(self):
        rng = np.random.default_rng(0)
        n = 5000
        scores = rng.uniform(0, 1, n)
        y = rng.binomial(1, 0.5, n)
        roc = _group_roc(scores, y)
        # Compute mean abs deviation of TPR from FPR (should be small)
        dev = np.mean(np.abs(roc[:, 0] - roc[:, 1]))
        assert dev < 0.05


class TestUpperHull:
    def test_hull_of_diagonal(self):
        # Points (0,0)-(0.5,0.5)-(1,1): hull is the same (collinear,
        # which counts as upper hull boundary)
        points = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        hull = _upper_hull(points)
        assert len(hull) >= 2  # at least endpoints

    def test_hull_above_diagonal(self):
        # Points above diagonal: should all stay on the hull
        points = np.array([[0, 0], [0.3, 0.8], [0.6, 0.9], [1, 1]])
        hull = _upper_hull(points)
        # Hull preserves the curve above
        assert len(hull) == 4


# =============================================================================
# Main algorithm
# =============================================================================

class TestEOHardt:
    @pytest.fixture
    def disparity_data(self):
        rng = np.random.default_rng(0)
        n = 30000
        a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
        y = np.zeros(n, dtype=int)
        for g, p in pi_true.items():
            mask = a == g
            y[mask] = rng.binomial(1, p, size=mask.sum())
        scores = 1.5 * y + 0.3 * (a - 1) + rng.normal(0, 0.5, n)
        return scores, y, a

    def test_runs_and_returns_thresholds(self, disparity_data):
        scores, y, a = disparity_data
        res = fit_eo_hardt(scores, a, y)
        assert set(res.thresholds_low.keys()) == set(np.unique(a).tolist())
        assert set(res.thresholds_high.keys()) == set(np.unique(a).tolist())
        # Lower threshold is always ≤ upper threshold
        for g in res.thresholds_low:
            assert res.thresholds_low[g] <= res.thresholds_high[g] + 1e-9

    def test_eod_substantially_reduced(self, disparity_data):
        scores, y, a = disparity_data
        res = fit_eo_hardt(scores, a, y)
        y_hat = predict_eo_hardt(scores, a, res, seed=0)
        eod = equalized_odds_difference(y_hat, y, a)
        # HPS-EO should reduce EOD significantly compared to unmitigated
        from ocf.postprocess import fit_unmitigated, predict_unmitigated
        unmit = fit_unmitigated(scores, y)
        y_hat_unmit = predict_unmitigated(scores, unmit)
        eod_unmit = equalized_odds_difference(y_hat_unmit, y, a)
        assert eod < eod_unmit

    def test_eod_near_zero_in_large_sample(self, disparity_data):
        scores, y, a = disparity_data
        res = fit_eo_hardt(scores, a, y)
        # Use a large number of random realizations to average over
        # randomization noise
        y_hats = [predict_eo_hardt(scores, a, res, seed=s) for s in range(20)]
        # Average TPR and FPR per group across realizations
        groups = list(np.unique(a))
        avg_tpr = {g: np.mean([true_positive_rate(yh, y, a, g) for yh in y_hats])
                   for g in groups}
        avg_fpr = {g: np.mean([false_positive_rate(yh, y, a, g) for yh in y_hats])
                   for g in groups}
        tpr_gap = max(avg_tpr.values()) - min(avg_tpr.values())
        fpr_gap = max(avg_fpr.values()) - min(avg_fpr.values())
        # Averaged over realizations, gaps should be small (<3%)
        assert tpr_gap < 0.03
        assert fpr_gap < 0.03

    def test_target_tpr_fpr_in_hull(self, disparity_data):
        scores, y, a = disparity_data
        res = fit_eo_hardt(scores, a, y)
        # Target point lies in [0,1]^2
        assert 0 <= res.target_tpr <= 1
        assert 0 <= res.target_fpr <= 1
        # Should be in the upper region: TPR > FPR for any useful classifier
        assert res.target_tpr > res.target_fpr

    def test_reproducibility_via_seed(self, disparity_data):
        scores, y, a = disparity_data
        res = fit_eo_hardt(scores, a, y)
        y1 = predict_eo_hardt(scores, a, res, seed=42)
        y2 = predict_eo_hardt(scores, a, res, seed=42)
        np.testing.assert_array_equal(y1, y2)

    def test_utility_options(self, disparity_data):
        scores, y, a = disparity_data
        for util in ["tpr_minus_fpr", "tpr_only", "min_loss"]:
            res = fit_eo_hardt(scores, a, y, utility=util)
            assert np.isfinite(res.target_tpr)
            assert np.isfinite(res.target_fpr)

    def test_unknown_utility_raises(self, disparity_data):
        scores, y, a = disparity_data
        with pytest.raises(ValueError):
            fit_eo_hardt(scores, a, y, utility="bogus")
