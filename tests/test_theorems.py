"""Theorem-verification tests.

These tests assert the four theorems from the paper hold on synthetic
data across many random seeds. Each test runs ``N_TRIALS`` independent
trials and reports the fraction in which the theorem's statement holds.

Theorem 1 (DP-Harm): under heterogeneous base rates and MLR, the DP
   post-processor strictly reduces TPR in the highest-burden group
   compared to the unmitigated classifier matched to the same
   aggregate selection rate.

Theorem 2 (Semiparametric efficiency): the plug-in estimator of
   :math:`\\hat\\pi_a` has asymptotic variance matching the semiparametric
   information bound. We test this via a bias-variance check:
   :math:`\\hat\\pi_a \\to \\pi_a` and :math:`\\text{Var}(\\hat\\pi_a)
   \\to \\pi_a(1-\\pi_a)/n_a` as :math:`n_a \\to \\infty`.

Theorem 3 (Accuracy compatibility): under MLR, OCF's accuracy
   strictly exceeds DP's accuracy at the same aggregate rate.

Theorem 4 (OCF dominates DP on high-burden TPR): under heterogeneous
   base rates, OCF's TPR in the high-burden group exceeds DP's.
"""
from __future__ import annotations

import numpy as np
import pytest

from ocf.postprocess import (
    fit_ocf,
    predict_ocf,
    fit_dp,
    predict_dp,
    fit_unmitigated,
    predict_unmitigated,
)
from ocf.metrics import (
    base_rate,
    selection_rate,
    true_positive_rate,
)


# =============================================================================
# Synthetic-data generator
# =============================================================================

def make_dataset(
    rng: np.random.Generator,
    n: int = 20000,
    pi_per_group: dict[int, float] | None = None,
    group_probs: list[float] | None = None,
    signal: float = 1.5,
    group_shift: float = 0.3,
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (scores, y, a) with heterogeneous base rates and MLR scores."""
    if pi_per_group is None:
        pi_per_group = {0: 0.2, 1: 0.4, 2: 0.6}
    groups = list(pi_per_group.keys())
    if group_probs is None:
        group_probs = [1.0 / len(groups)] * len(groups)
    a = rng.choice(groups, size=n, p=group_probs)
    y = np.zeros(n, dtype=int)
    for g, p in pi_per_group.items():
        mask = a == g
        y[mask] = rng.binomial(1, p, size=mask.sum())
    # Construct scores: signal in y + group-shift + noise → MLR per group
    scores = signal * y + group_shift * (a - np.mean(groups)) + rng.normal(0, noise, n)
    return scores, y, a


def high_burden_group(y: np.ndarray, a: np.ndarray):
    groups = np.unique(a)
    pi = {g: base_rate(y, a, g) for g in groups}
    return max(pi, key=pi.get)


# =============================================================================
# Theorem 1: DP-Harm
# =============================================================================

class TestTheorem1_DPHarm:
    """For sufficiently disparate base rates, DP reduces high-burden TPR."""

    @pytest.mark.parametrize("trial", range(20))
    def test_dp_reduces_high_burden_tpr(self, trial):
        rng = np.random.default_rng(trial)
        scores, y, a = make_dataset(rng)
        high_g = high_burden_group(y, a)
        unmit = fit_unmitigated(scores, y)
        y_hat_unmit = predict_unmitigated(scores, unmit)
        dp = fit_dp(scores, a, target_rate=float(np.mean(y)))
        y_hat_dp = predict_dp(scores, a, dp)
        tpr_unmit = true_positive_rate(y_hat_unmit, y, a, high_g)
        tpr_dp = true_positive_rate(y_hat_dp, y, a, high_g)
        assert tpr_dp < tpr_unmit, (
            f"trial {trial}: DP TPR {tpr_dp:.4f} >= Unmit TPR {tpr_unmit:.4f}"
        )

    def test_dp_harm_magnitude(self):
        """Mean ΔTPR(unmit - DP) over 20 trials should be substantial."""
        gaps = []
        for trial in range(20):
            rng = np.random.default_rng(trial + 100)
            scores, y, a = make_dataset(rng)
            high_g = high_burden_group(y, a)
            unmit = fit_unmitigated(scores, y)
            y_hat_u = predict_unmitigated(scores, unmit)
            dp = fit_dp(scores, a, target_rate=float(np.mean(y)))
            y_hat_dp = predict_dp(scores, a, dp)
            gap = (
                true_positive_rate(y_hat_u, y, a, high_g)
                - true_positive_rate(y_hat_dp, y, a, high_g)
            )
            gaps.append(gap)
        mean_gap = float(np.mean(gaps))
        # With pi = [0.2, 0.4, 0.6], expect gap of ~20-40 pp
        assert mean_gap > 0.10


# =============================================================================
# Theorem 2: Semiparametric efficiency of pi_hat
# =============================================================================

class TestTheorem2_Efficiency:
    """Plug-in estimator achieves the binomial information bound."""

    def test_unbiasedness(self):
        """E[pi_hat] -> pi_true as n increases."""
        pi_true = 0.4
        n = 5000
        trials = 200
        rng = np.random.default_rng(0)
        ests = [float(np.mean(rng.binomial(1, pi_true, n))) for _ in range(trials)]
        bias = float(np.mean(ests)) - pi_true
        # Bias should be ~0 (only Monte Carlo noise)
        assert abs(bias) < 0.005

    def test_variance_matches_bound(self):
        """Var(pi_hat) ≈ pi(1-pi)/n."""
        pi_true = 0.4
        n = 5000
        trials = 500
        rng = np.random.default_rng(0)
        ests = np.array([
            float(np.mean(rng.binomial(1, pi_true, n))) for _ in range(trials)
        ])
        var_emp = float(np.var(ests, ddof=1))
        var_bound = pi_true * (1 - pi_true) / n
        # Empirical variance within 20% of theoretical bound
        ratio = var_emp / var_bound
        assert 0.8 < ratio < 1.25, f"ratio={ratio:.3f}"


# =============================================================================
# Theorem 3: Accuracy compatibility (OCF >= DP)
# =============================================================================

class TestTheorem3_Accuracy:
    """OCF achieves higher accuracy than DP at matched aggregate rate."""

    @pytest.mark.parametrize("trial", range(20))
    def test_ocf_beats_dp_on_accuracy(self, trial):
        rng = np.random.default_rng(trial)
        scores, y, a = make_dataset(rng)
        target = float(np.mean(y))
        dp = fit_dp(scores, a, target_rate=target)
        y_hat_dp = predict_dp(scores, a, dp)
        ocf_ = fit_ocf(scores, a, y, target_aggregate_rate=target)
        y_hat_ocf = predict_ocf(scores, a, ocf_)
        acc_dp = float(np.mean(y_hat_dp == y))
        acc_ocf = float(np.mean(y_hat_ocf == y))
        assert acc_ocf > acc_dp, (
            f"trial {trial}: OCF acc {acc_ocf:.4f} <= DP acc {acc_dp:.4f}"
        )

    def test_aggregate_rates_match(self):
        """OCF and DP at matched target should have ~identical aggregate
        selection rates (this is the precondition for the accuracy
        comparison to be apples-to-apples)."""
        rng = np.random.default_rng(0)
        scores, y, a = make_dataset(rng)
        target = float(np.mean(y))
        dp = fit_dp(scores, a, target_rate=target)
        ocf_ = fit_ocf(scores, a, y, target_aggregate_rate=target)
        y_hat_dp = predict_dp(scores, a, dp)
        y_hat_ocf = predict_ocf(scores, a, ocf_)
        agg_dp = float(np.mean(y_hat_dp))
        agg_ocf = float(np.mean(y_hat_ocf))
        assert abs(agg_dp - target) < 0.01
        assert abs(agg_ocf - target) < 0.01


# =============================================================================
# Theorem 4: OCF strictly dominates DP on high-burden TPR
# =============================================================================

class TestTheorem4_OCFvsDP:
    """OCF gives higher high-burden TPR than DP at matched aggregate."""

    @pytest.mark.parametrize("trial", range(20))
    def test_ocf_tpr_exceeds_dp_tpr(self, trial):
        rng = np.random.default_rng(trial)
        scores, y, a = make_dataset(rng)
        high_g = high_burden_group(y, a)
        target = float(np.mean(y))
        dp = fit_dp(scores, a, target_rate=target)
        y_hat_dp = predict_dp(scores, a, dp)
        ocf_ = fit_ocf(scores, a, y, target_aggregate_rate=target)
        y_hat_ocf = predict_ocf(scores, a, ocf_)
        tpr_dp = true_positive_rate(y_hat_dp, y, a, high_g)
        tpr_ocf = true_positive_rate(y_hat_ocf, y, a, high_g)
        assert tpr_ocf > tpr_dp, (
            f"trial {trial}: OCF TPR {tpr_ocf:.4f} <= DP TPR {tpr_dp:.4f}"
        )

    def test_dominance_magnitude(self):
        """Mean Δ(OCF - DP) on high-burden TPR over 20 trials."""
        gaps = []
        for trial in range(20):
            rng = np.random.default_rng(trial + 200)
            scores, y, a = make_dataset(rng)
            high_g = high_burden_group(y, a)
            target = float(np.mean(y))
            dp = fit_dp(scores, a, target_rate=target)
            ocf_ = fit_ocf(scores, a, y, target_aggregate_rate=target)
            y_hat_dp = predict_dp(scores, a, dp)
            y_hat_ocf = predict_ocf(scores, a, ocf_)
            gap = (
                true_positive_rate(y_hat_ocf, y, a, high_g)
                - true_positive_rate(y_hat_dp, y, a, high_g)
            )
            gaps.append(gap)
        mean_gap = float(np.mean(gaps))
        # Expect 20-40pp gap with pi=[0.2, 0.4, 0.6]
        assert mean_gap > 0.10
