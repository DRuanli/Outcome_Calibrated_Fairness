"""Tests for ``ocf.theory`` — MLR verification and Bayes-optimal rates."""
from __future__ import annotations

import numpy as np
import pytest

from ocf.theory import (
    estimate_likelihood_ratio,
    verify_mlr,
    bayes_optimal_rate,
    bayes_optimal_rates_per_group,
)


class TestLikelihoodRatio:
    def test_positive_signal_increases_lr(self):
        # Scores have positive signal in y: LR should be monotone non-decreasing
        rng = np.random.default_rng(0)
        n = 20000
        y = rng.binomial(1, 0.4, n)
        scores = 1.5 * y + rng.normal(0, 0.5, n)
        centers, lrs, _ = estimate_likelihood_ratio(scores, y, n_bins=10)
        # LR should generally increase with score (MLR holds)
        diffs = np.diff(lrs)
        # Allow some bins to have small negative diffs (estimation noise)
        assert int((diffs > 0).sum()) > len(diffs) * 0.6


class TestMLRVerification:
    def test_mlr_holds_for_logistic_scores(self):
        rng = np.random.default_rng(0)
        n = 20000
        a = rng.choice([0, 1, 2], size=n)
        y = rng.binomial(1, 0.3, n)
        # Score: linear in y → MLR
        scores = 1.5 * y + rng.normal(0, 0.5, n)
        result = verify_mlr(scores, a, y, n_bins=10, tolerance=0.05)
        # MLR should hold for most groups when tolerance is reasonable
        n_holding = sum(1 for r in result.values() if r["mlr_holds"])
        assert n_holding >= 2

    def test_mlr_returns_one_dict_per_group(self):
        rng = np.random.default_rng(0)
        n = 5000
        a = rng.choice(["X", "Y", "Z"], size=n)
        y = rng.binomial(1, 0.4, n)
        scores = rng.uniform(0, 1, n)
        result = verify_mlr(scores, a, y, n_bins=8)
        assert set(result.keys()) == {"X", "Y", "Z"}
        for g, r in result.items():
            assert "mlr_holds" in r
            assert "lr_values" in r
            assert "score_centers" in r


class TestBayesOptimal:
    def test_bayes_threshold_and_rate_finite(self):
        rng = np.random.default_rng(0)
        n = 5000
        y = rng.binomial(1, 0.4, n)
        scores = 1.5 * y + rng.normal(0, 0.5, n)
        t, rate = bayes_optimal_rate(scores, y)
        assert np.isfinite(t)
        assert 0 < rate < 1

    def test_bayes_rate_increases_with_pi(self):
        """Higher base rate → higher Bayes-optimal selection rate (Lemma 3.2)."""
        rng = np.random.default_rng(0)
        n = 10000
        rates = {}
        for pi in [0.2, 0.4, 0.6]:
            y = rng.binomial(1, pi, n)
            scores = 1.5 * y + rng.normal(0, 0.5, n)
            _, sel = bayes_optimal_rate(scores, y)
            rates[pi] = sel
        # Monotone in pi
        assert rates[0.2] < rates[0.4] < rates[0.6]

    def test_per_group_dict_shape(self):
        rng = np.random.default_rng(0)
        n = 6000
        a = rng.choice([0, 1, 2], size=n)
        y = rng.binomial(1, 0.4, n)
        scores = 1.5 * y + rng.normal(0, 0.5, n)
        out = bayes_optimal_rates_per_group(scores, a, y)
        for g in [0, 1, 2]:
            assert g in out
            for key in ["threshold", "rho_bayes", "pi"]:
                assert key in out[g]
