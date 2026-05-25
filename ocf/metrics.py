"""
ocf.metrics — group-wise rates and fairness metrics.

Every function operates on flat numpy arrays. Predictions ``y_hat`` and
labels ``y`` are 0/1 integers; ``a`` is the protected attribute (any hashable).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# =============================================================================
# Basic group-wise rates
# =============================================================================

def base_rate(y: np.ndarray, a: np.ndarray, group: Any) -> float:
    r"""Plug-in estimator of :math:`\pi_a = P(Y=1 \mid A=a)`.

    Returns ``nan`` if no observation in ``group``.
    """
    mask = np.asarray(a) == group
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(y)[mask]))


def selection_rate(y_hat: np.ndarray, a: np.ndarray, group: Any) -> float:
    r"""Group selection rate :math:`\rho_a = P(\hat Y = 1 \mid A=a)`."""
    mask = np.asarray(a) == group
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(y_hat)[mask]))


def true_positive_rate(
    y_hat: np.ndarray, y: np.ndarray, a: np.ndarray, group: Any
) -> float:
    r"""Within-group TPR :math:`\tau_a = P(\hat Y = 1 \mid Y=1, A=a)`."""
    mask = (np.asarray(a) == group) & (np.asarray(y) == 1)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(y_hat)[mask]))


def false_positive_rate(
    y_hat: np.ndarray, y: np.ndarray, a: np.ndarray, group: Any
) -> float:
    r"""Within-group FPR :math:`\nu_a = P(\hat Y = 1 \mid Y=0, A=a)`."""
    mask = (np.asarray(a) == group) & (np.asarray(y) == 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(y_hat)[mask]))


def positive_predictive_value(
    y_hat: np.ndarray, y: np.ndarray, a: np.ndarray, group: Any
) -> float:
    r"""Within-group PPV :math:`P(Y=1 \mid \hat Y = 1, A=a)`."""
    mask = (np.asarray(a) == group) & (np.asarray(y_hat) == 1)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(y)[mask]))


# =============================================================================
# Standard group-fairness metrics
# =============================================================================

def demographic_parity_difference(
    y_hat: np.ndarray, a: np.ndarray, groups: Sequence[Any] | None = None
) -> float:
    r"""Demographic parity difference :math:`\max_a \rho_a - \min_a \rho_a`."""
    if groups is None:
        groups = list(np.unique(a))
    rates = [selection_rate(y_hat, a, g) for g in groups]
    finite = [r for r in rates if np.isfinite(r)]
    if not finite:
        return float("nan")
    return float(max(finite) - min(finite))


def equalized_odds_difference(
    y_hat: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    groups: Sequence[Any] | None = None,
) -> float:
    r"""Equalized-odds difference :math:`\max(\Delta\tau, \Delta\nu)`.

    Defined as the larger of the across-group spread in TPR and the
    across-group spread in FPR. This is the same definition used by
    Fairlearn's ``equalized_odds_difference``.
    """
    if groups is None:
        groups = list(np.unique(a))
    tprs = [true_positive_rate(y_hat, y, a, g) for g in groups]
    fprs = [false_positive_rate(y_hat, y, a, g) for g in groups]
    tprs_f = [r for r in tprs if np.isfinite(r)]
    fprs_f = [r for r in fprs if np.isfinite(r)]
    if not tprs_f or not fprs_f:
        return float("nan")
    return float(max(max(tprs_f) - min(tprs_f), max(fprs_f) - min(fprs_f)))


# =============================================================================
# OCF violation: two definitions
# =============================================================================

def _optimal_c(
    rho: np.ndarray, pi: np.ndarray, w: np.ndarray | None = None
) -> float:
    r"""Find :math:`c^* = \arg\min_c \max_a |\rho_a - c \pi_a|`.

    The L_inf objective is convex piecewise-linear in c with a single
    minimum; we locate it by a 1-D golden-section-like search after
    finding the bracketing interval. For the common case of 2-6 groups,
    this is fast (<1ms).

    If ``w`` is given (group weights), the L_2 closed-form
    :math:`c = \sum_a w_a \pi_a \rho_a / \sum_a w_a \pi_a^2` is returned
    instead; the L_inf optimum is similar but not identical for very
    unbalanced groups.
    """
    rho = np.asarray(rho, dtype=float)
    pi = np.asarray(pi, dtype=float)
    valid = (pi > 0) & np.isfinite(rho) & np.isfinite(pi)
    if not valid.any():
        return float("nan")
    rho = rho[valid]
    pi = pi[valid]

    if w is not None:
        w = np.asarray(w, dtype=float)[valid]
        num = float(np.sum(w * pi * rho))
        den = float(np.sum(w * pi * pi))
        if den <= 0:
            return float("nan")
        return num / den

    # L_inf: search over candidates. Optimum lies in
    # [min(rho/pi), max(rho/pi)]. Evaluate on a 256-point grid + refine.
    ratios = rho / pi
    lo, hi = float(np.min(ratios)), float(np.max(ratios))
    if lo == hi:
        return float(lo)

    def obj(c: float) -> float:
        return float(np.max(np.abs(rho - c * pi)))

    # Golden-section
    invphi = (np.sqrt(5) - 1) / 2  # 1/golden_ratio
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/golden_ratio^2
    a_, b_ = lo, hi
    h = b_ - a_
    n = int(np.ceil(np.log(1e-12 / h) / np.log(invphi)))
    c1 = a_ + invphi2 * h
    c2 = a_ + invphi * h
    f1, f2 = obj(c1), obj(c2)
    for _ in range(n):
        if f1 < f2:
            b_, c2, f2 = c2, c1, f1
            h = invphi * h
            c1 = a_ + invphi2 * h
            f1 = obj(c1)
        else:
            a_, c1, f1 = c1, c2, f2
            h = invphi * h
            c2 = a_ + invphi * h
            f2 = obj(c2)
    return (a_ + b_) / 2


def ocf_violation(
    y_hat: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    groups: Sequence[Any] | None = None,
    c: float | None = None,
    normalize: bool = True,
) -> float:
    r"""OCF violation metric (paper Definition 3.6, :math:`L^\infty` form).

    Defined as

    .. math::
        \mathrm{OCFViol}(\hat Y) =
        \frac{\min_{c \geq 0}\, \max_a |\rho_a - c \pi_a|}
             {\max_a \pi_a}.

    Returns a value in :math:`[0, 1]`. An optimally OCF-fair classifier
    has ``OCFViol == 0`` (up to plug-in noise in :math:`\hat\pi_a`).

    Parameters
    ----------
    y_hat, y, a : arrays
        Predicted labels, true labels, and protected attribute.
    groups : sequence, optional
        Order of groups (default: ``np.unique(a)``).
    c : float, optional
        If supplied, do *not* optimize over c; report the absolute
        deviation under the given c. Useful for diagnostic auditing of
        a deployed OCF classifier where the fitted c is known.
    normalize : bool, default True
        If False, return the raw L_inf deviation (in units of selection
        rate) without dividing by ``max(pi_a)``.
    """
    if groups is None:
        groups = list(np.unique(a))
    pi = np.array([base_rate(y, a, g) for g in groups])
    rho = np.array([selection_rate(y_hat, a, g) for g in groups])

    if c is None:
        c = _optimal_c(rho, pi)
    if not np.isfinite(c):
        return float("nan")

    raw = float(np.max(np.abs(rho - c * pi)))
    if not normalize:
        return raw
    max_pi = float(np.nanmax(pi))
    if max_pi <= 0:
        return float("nan")
    return raw / max_pi


def ocf_violation_ratio_spread(
    y_hat: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    groups: Sequence[Any] | None = None,
) -> float:
    r"""Legacy/diagnostic OCF violation: spread of ratios
    :math:`\max_a \rho_a/\pi_a - \min_a \rho_a/\pi_a`.

    Retained for backward compatibility with the original code base and
    for diagnostic intuition (it answers "by how much do the screening-
    multipliers differ across groups?"). Paper headline numbers use
    :func:`ocf_violation` instead.
    """
    if groups is None:
        groups = list(np.unique(a))
    ratios: list[float] = []
    for g in groups:
        pi_g = base_rate(y, a, g)
        if pi_g is None or not np.isfinite(pi_g) or pi_g <= 0:
            continue
        rho_g = selection_rate(y_hat, a, g)
        if not np.isfinite(rho_g):
            continue
        ratios.append(rho_g / pi_g)
    if not ratios:
        return float("nan")
    return float(max(ratios) - min(ratios))


# =============================================================================
# Calibration disparity (diagnostic)
# =============================================================================

def calibration_disparity(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    n_bins: int = 10,
    min_bin_size: int = 30,
) -> float:
    """Maximum across-group disparity in observed positive rate within
    score deciles. Lower values indicate better cross-group calibration.

    Only bins where every group has at least ``min_bin_size`` observations
    are scored; bins with insufficient data are skipped.
    """
    scores = np.asarray(scores)
    y = np.asarray(y)
    a = np.asarray(a)
    groups = np.unique(a)
    bins = np.linspace(0, 1, n_bins + 1)
    max_disp = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        obs: list[float] = []
        for g in groups:
            mask = (a == g) & (scores >= lo) & (scores < hi)
            if int(mask.sum()) < min_bin_size:
                continue
            obs.append(float(np.mean(y[mask])))
        if len(obs) >= 2:
            max_disp = max(max_disp, float(max(obs) - min(obs)))
    return float(max_disp)
