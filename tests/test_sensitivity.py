"""Tests for ``ocf.sensitivity``."""
from __future__ import annotations

import numpy as np
import pytest

from ocf.sensitivity import sensitivity_sweep, find_c_matching_metric


@pytest.fixture
def disparity_data():
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    pi_true = {0: 0.2, 1: 0.4, 2: 0.6}
    y = np.zeros(n, dtype=int)
    for g, p in pi_true.items():
        mask = a == g
        y[mask] = rng.binomial(1, p, size=mask.sum())
    scores = 1.5 * y + 0.3 * (a - 1) + rng.normal(0, 0.5, n)
    return scores, y, a, pi_true


class TestSensitivitySweep:
    def test_sweep_returns_correct_number_of_rows(self, disparity_data):
        scores, y, a, _ = disparity_data
        c_values = [0.5, 0.75, 1.0, 1.25, 1.5]
        rows = sensitivity_sweep(scores, y, a, c_values=c_values)
        assert len(rows) == 5

    def test_at_c_equals_1_matches_default_ocf(self, disparity_data):
        scores, y, a, pi_true = disparity_data
        rows = sensitivity_sweep(scores, y, a, c_values=[1.0])
        row = rows[0]
        # At c=1.0, OCF has rho_g ≈ pi_g for each group
        for g, pi in pi_true.items():
            assert row[f"rho_{g}"] == pytest.approx(pi, abs=0.02)

    def test_higher_c_increases_aggregate_selection(self, disparity_data):
        scores, y, a, _ = disparity_data
        c_values = [0.5, 1.0, 1.5]
        rows = sensitivity_sweep(scores, y, a, c_values=c_values)
        # Aggregate selection ≈ c * mean(y) — monotone increasing in c
        sel_means = []
        for r in rows:
            # Reconstruct via sum of rho_g * P(A=g); easier: from groups present
            groups_in_row = [k for k in r if k.startswith("rho_")]
            sel = float(np.sum(np.array([r[k] for k in groups_in_row])
                              * 1.0 / len(groups_in_row)))
            sel_means.append(sel)
        # Selection rate at higher c should exceed selection at lower c
        assert sel_means[0] < sel_means[1] < sel_means[2]

    def test_high_burden_shortcut(self, disparity_data):
        scores, y, a, _ = disparity_data
        rows = sensitivity_sweep(scores, y, a, c_values=[1.0],
                                high_burden_group=2)
        assert "TPR_high_burden" in rows[0]
        assert "rho_high_burden" in rows[0]


class TestBisectionSearch:
    def test_recover_target_aggregate(self, disparity_data):
        scores, y, a, _ = disparity_data
        target = 0.4
        c = find_c_matching_metric(scores, y, a,
                                   target_metric="aggregate_selection_rate",
                                   target_value=target,
                                   tol=1e-3)
        # Verify by running OCF with that c
        from ocf.postprocess import fit_ocf, predict_ocf
        fit = fit_ocf(scores, a, y, c=c)
        y_hat = predict_ocf(scores, a, fit)
        agg = float(np.mean(y_hat))
        assert agg == pytest.approx(target, abs=0.01)

    def test_recover_target_tpr(self, disparity_data):
        scores, y, a, _ = disparity_data
        target = 0.8
        c = find_c_matching_metric(scores, y, a,
                                   target_metric="high_burden_tpr",
                                   target_value=target,
                                   tol=1e-3)
        from ocf.postprocess import fit_ocf, predict_ocf
        from ocf.metrics import true_positive_rate
        fit = fit_ocf(scores, a, y, c=c)
        y_hat = predict_ocf(scores, a, fit)
        tpr = true_positive_rate(y_hat, y, a, 2)
        assert tpr == pytest.approx(target, abs=0.02)
