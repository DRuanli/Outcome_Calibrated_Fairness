"""
ocf.theory — empirical verification of theoretical assumptions.

Provides utilities to verify Assumption A2′ (Monotone Likelihood Ratio,
within group) and compute Bayes-optimal within-group thresholds and
selection rates, used by Theorem 3 and Lemma 3.1 of the paper.

These functions are diagnostic: a deployed OCF system does not require
them to run. They are used by the unit tests and by
``scripts/verify_mlr.py`` to produce the empirical-robustness table
reported in Section 3.7 of the paper.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def estimate_likelihood_ratio(
    scores: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    min_count_per_bin: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Estimate the within-group likelihood ratio
    :math:`\mathrm{LR}(v) = f_{s|Y=1}(v) / f_{s|Y=0}(v)` by binning.

    Parameters
    ----------
    scores, y : arrays of equal length
        Scores and binary labels for a single group.
    n_bins : int
        Number of quantile-based bins of the score.
    min_count_per_bin : int
        Bins where either pos or neg counts are below this threshold
        are dropped (estimator unstable).

    Returns
    -------
    bin_centers : array
        Score quantile bin centers.
    lr_estimates : array
        Estimated LR at each bin center.
    bin_counts : array, shape (n_bins, 2)
        Per-bin (n_pos, n_neg) counts (after filtering).
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([]), np.zeros((0, 2), dtype=int)

    # Quantile-based bins
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(scores, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return np.array([]), np.array([]), np.zeros((0, 2), dtype=int)
    bin_idx = np.clip(np.digitize(scores, bin_edges) - 1, 0, len(bin_edges) - 2)

    lr_vals: list[float] = []
    centers: list[float] = []
    counts: list[tuple[int, int]] = []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        np_b = int((y[mask] == 1).sum())
        nn_b = int((y[mask] == 0).sum())
        if np_b < min_count_per_bin or nn_b < min_count_per_bin:
            continue
        # density estimate per bin: (count / total) / bin width
        width = bin_edges[b + 1] - bin_edges[b]
        if width <= 0:
            continue
        f1 = (np_b / n_pos) / width
        f0 = (nn_b / n_neg) / width
        if f0 <= 0:
            continue
        lr_vals.append(f1 / f0)
        centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        counts.append((np_b, nn_b))

    return np.array(centers), np.array(lr_vals), np.array(counts)


def verify_mlr(
    scores: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    tolerance: float = 0.0,
) -> dict[Any, dict[str, Any]]:
    r"""Test whether MLR (Assumption A2′) holds empirically for each group.

    For each group, estimates the likelihood ratio at ``n_bins``
    quantile-based bin centers and checks if it is monotone
    non-decreasing in the score (up to ``tolerance``).

    Parameters
    ----------
    scores, a, y : arrays
        Scores, protected attribute, labels.
    n_bins : int
        Number of quantile bins.
    tolerance : float, default 0.0
        Allow LR to *decrease* by up to this much between adjacent bins
        without flagging violation. Default 0 = strict monotonicity test.

    Returns
    -------
    dict
        For each group ``g``, a dict with:

        - ``mlr_holds`` : bool
        - ``n_violations`` : int (number of decreasing pairs)
        - ``max_decrease`` : float (largest decrease in LR observed)
        - ``lr_values`` : np.ndarray (estimated LRs)
        - ``score_centers`` : np.ndarray (bin centers)
    """
    out: dict[Any, dict[str, Any]] = {}
    a = np.asarray(a)
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    for g in np.unique(a):
        mask = a == g
        centers, lrs, counts = estimate_likelihood_ratio(
            scores[mask], y[mask], n_bins=n_bins
        )
        if len(lrs) < 2:
            out[g] = {
                "mlr_holds": False,
                "n_violations": -1,
                "max_decrease": np.nan,
                "lr_values": lrs,
                "score_centers": centers,
                "reason": "insufficient data for binning",
            }
            continue
        diffs = np.diff(lrs)
        n_viol = int((diffs < -tolerance).sum())
        max_dec = float(-np.min(diffs))
        out[g] = {
            "mlr_holds": bool(n_viol == 0),
            "n_violations": n_viol,
            "max_decrease": max_dec,
            "lr_values": lrs,
            "score_centers": centers,
        }
    return out


def bayes_optimal_rate(
    scores: np.ndarray, y: np.ndarray, n_thresholds: int = 200
) -> tuple[float, float]:
    r"""Within-group Bayes-optimal threshold and selection rate.

    For a single group, sweeps thresholds and returns the one
    maximizing within-group accuracy
    :math:`\pi \tau + (1-\pi)(1-\nu)`, along with the corresponding
    selection rate.

    Returns
    -------
    threshold, selection_rate : float, float
        :math:`t_a^{\mathrm{Bayes}}` and
        :math:`\rho_a^{\mathrm{Bayes}}`.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    if len(scores) == 0:
        return float("nan"), float("nan")
    pi = float(np.mean(y))
    # Candidates: quantile-spaced thresholds over the score distribution
    qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    candidates = np.quantile(scores, qs)
    best_t = float(np.median(scores))
    best_acc = -np.inf
    for t in candidates:
        y_hat = (scores >= t).astype(int)
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue
        tpr = float(np.mean(y_hat[y == 1]))
        fpr = float(np.mean(y_hat[y == 0]))
        acc = pi * tpr + (1 - pi) * (1 - fpr)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    sel = float(np.mean(scores >= best_t))
    return best_t, sel


def bayes_optimal_rates_per_group(
    scores: np.ndarray, a: np.ndarray, y: np.ndarray
) -> dict[Any, dict[str, float]]:
    """Return per-group :math:`(t_a^{\\mathrm{Bayes}},
    \\rho_a^{\\mathrm{Bayes}})` and base rate :math:`\\pi_a`."""
    out: dict[Any, dict[str, float]] = {}
    a = np.asarray(a)
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    for g in np.unique(a):
        mask = a == g
        t_g, rho_g = bayes_optimal_rate(scores[mask], y[mask])
        out[g] = {
            "threshold": t_g,
            "rho_bayes": rho_g,
            "pi": float(np.mean(y[mask])) if mask.any() else float("nan"),
        }
    return out
