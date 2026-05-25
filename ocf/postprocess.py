"""
ocf.postprocess — post-processing algorithms.

This module implements:

- **Algorithm 1**: OCF post-processing (the paper's contribution).
- **Algorithm 2**: Demographic-parity (DP) post-processing, i.e.,
  Hardt-Price-Srebro restricted to threshold rules.
- **Algorithm 4**: Group-wise isotonic calibration baseline.
- **Unmitigated baseline**: single global threshold.

For the reference Hardt-Price-Srebro Equalized-Odds post-processor
(Algorithm 3 with randomization), see :mod:`ocf.eo_hardt`.

Every ``fit_*`` returns a result dataclass; the corresponding
``predict_*`` consumes it. Train/test split must be honored by the
caller (i.e., fit on validation, predict on test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ocf.metrics import base_rate, selection_rate

# =============================================================================
# Utilities
# =============================================================================

def _quantile_threshold(scores: np.ndarray, target_rate: float) -> float:
    r"""Threshold ``t`` such that the empirical
    :math:`P(\text{scores} \geq t) \approx \text{target\_rate}`.

    Convention: ``target_rate <= 0`` → ``+inf`` (predict no one),
    ``target_rate >= 1`` → ``-inf`` (predict everyone). For interior
    targets, ``np.quantile(scores, 1 - target_rate)`` is used.
    """
    if target_rate <= 0:
        return float("inf")
    if target_rate >= 1:
        return float("-inf")
    return float(np.quantile(scores, 1.0 - target_rate))


def _safe_clip(rate: float, lo: float = 1e-4, hi: float = 1 - 1e-4) -> float:
    """Clip a probability to numerically safe range."""
    return float(min(max(rate, lo), hi))


def apply_per_group_thresholds(
    scores: np.ndarray,
    a: np.ndarray,
    thresholds: Mapping[Any, float],
) -> np.ndarray:
    r"""Apply group-conditional thresholds:
    :math:`\hat Y = \mathbb{1}\{s(X) \geq t_A\}`.

    Groups present in ``a`` but absent from ``thresholds`` are predicted
    negative (threshold ``+inf``).
    """
    scores = np.asarray(scores)
    a = np.asarray(a)
    y_hat = np.zeros_like(scores, dtype=int)
    for g, t in thresholds.items():
        mask = a == g
        if not mask.any():
            continue
        y_hat[mask] = (scores[mask] >= t).astype(int)
    return y_hat


# =============================================================================
# Algorithm 1: OCF post-processing
# =============================================================================

@dataclass
class OCFResult:
    """Output of :func:`fit_ocf`.

    Attributes
    ----------
    thresholds : dict
        Per-group thresholds :math:`\\{t_a\\}`.
    c : float
        The screening multiplier used (input or auto-fit).
    pi_hat : dict
        Per-group plug-in estimates :math:`\\hat\\pi_a`.
    target_rho : dict
        Per-group target selection rates :math:`c \\cdot \\hat\\pi_a`,
        possibly clipped to ``(0, 1)``.
    target_aggregate_rate : float
        Aggregate selection rate the multiplier was tuned to (only
        meaningful when ``c`` was auto-fit).
    """
    thresholds: dict[Any, float]
    c: float
    pi_hat: dict[Any, float]
    target_rho: dict[Any, float]
    target_aggregate_rate: float = float("nan")


def fit_ocf(
    scores: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    c: float | None = None,
    target_aggregate_rate: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> OCFResult:
    r"""Algorithm 1: fit OCF group-conditional thresholds.

    Given predicted scores on a validation set, choose per-group
    thresholds so that the resulting classifier satisfies the OCF
    criterion :math:`\rho_a = c \cdot \pi_a` for all :math:`a`.

    Parameters
    ----------
    scores : array of shape (n,)
        Predicted :math:`P(Y=1 \mid X)` from a trained classifier,
        evaluated on the validation set.
    a : array of shape (n,)
        Protected attribute.
    y : array of shape (n,)
        Binary labels (0/1) on the validation set.
    c : float, optional
        Screening multiplier. If ``None``, ``c`` is chosen so the
        aggregate selection rate equals ``target_aggregate_rate``
        (or ``mean(y)`` if that is also ``None``).
    target_aggregate_rate : float, optional
        Aggregate selection rate to match when ``c`` is auto-fit.
    sample_weight : array, optional
        Survey/sample weights. If supplied, all of :math:`\hat\pi_a`,
        the implied :math:`\bar\pi`, and ``target_aggregate_rate``
        defaults are computed as weighted means.

    Returns
    -------
    OCFResult
        See :class:`OCFResult`.

    Notes
    -----
    With the default settings (``c=None``, ``target_aggregate_rate=None``,
    ``sample_weight=None``), the fitted ``c`` satisfies

    .. math::
       c \cdot \sum_a w_a \hat\pi_a = \mathrm{mean}(y),

    where :math:`w_a = P(A=a)`. By the moment identity
    :math:`\mathrm{mean}(y) = \sum_a w_a \hat\pi_a`, this gives
    :math:`c = 1` exactly, so the default OCF classifier matches the
    "screening rate equals base rate" interpretation. Pass a non-default
    ``target_aggregate_rate`` to choose any other multiplier.

    Complexity is :math:`O(n)` to estimate :math:`\hat\pi_a` plus
    :math:`O(n_a \log n_a)` per group for threshold quantile lookup.
    """
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    y = np.asarray(y).astype(int)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    groups = list(np.unique(a))

    # Step 1: estimate per-group base rates
    pi_hat: dict[Any, float] = {}
    w_group: dict[Any, float] = {}
    for g in groups:
        mask = a == g
        if not mask.any():
            pi_hat[g] = float("nan")
            w_group[g] = 0.0
            continue
        if sample_weight is None:
            pi_hat[g] = float(np.mean(y[mask]))
            w_group[g] = float(mask.sum()) / float(len(a))
        else:
            sw = sample_weight[mask]
            sw_sum = float(np.sum(sw))
            pi_hat[g] = (
                float(np.sum(sw * y[mask]) / sw_sum) if sw_sum > 0 else float("nan")
            )
            total_w = float(np.sum(sample_weight))
            w_group[g] = sw_sum / total_w if total_w > 0 else 0.0

    # Step 2: choose c (if not given) so aggregate selection rate
    # matches target.
    if c is None:
        if target_aggregate_rate is None:
            if sample_weight is None:
                target_aggregate_rate = float(np.mean(y))
            else:
                wsum = float(np.sum(sample_weight))
                target_aggregate_rate = (
                    float(np.sum(sample_weight * y) / wsum)
                    if wsum > 0
                    else float("nan")
                )
        # Weighted mean of pi over groups (this equals mean(y) by the
        # tower-law identity, so the default gives c = 1; we compute it
        # explicitly to support arbitrary target_aggregate_rate inputs).
        expected_pi = sum(
            w_group[g] * pi_hat[g] for g in groups if np.isfinite(pi_hat[g])
        )
        if expected_pi <= 0:
            raise ValueError(
                "All group base rates are zero or non-finite; "
                "cannot fit OCF."
            )
        c = float(target_aggregate_rate) / expected_pi
    else:
        c = float(c)
        if target_aggregate_rate is None:
            target_aggregate_rate = c * sum(
                w_group[g] * pi_hat[g] for g in groups if np.isfinite(pi_hat[g])
            )

    # Step 3: compute target selection rates per group
    target_rho = {
        g: _safe_clip(c * pi_hat[g]) if np.isfinite(pi_hat[g]) else 0.0
        for g in groups
    }

    # Step 4: solve threshold per group
    thresholds: dict[Any, float] = {}
    for g in groups:
        scores_g = scores[a == g]
        if len(scores_g) == 0:
            thresholds[g] = float("inf")
        else:
            thresholds[g] = _quantile_threshold(scores_g, target_rho[g])

    return OCFResult(
        thresholds=thresholds,
        c=c,
        pi_hat=pi_hat,
        target_rho=target_rho,
        target_aggregate_rate=float(target_aggregate_rate),
    )


def predict_ocf(
    scores: np.ndarray, a: np.ndarray, result: OCFResult
) -> np.ndarray:
    """Apply fitted OCF thresholds to produce binary predictions."""
    return apply_per_group_thresholds(scores, a, result.thresholds)


# =============================================================================
# Algorithm 2: Demographic Parity post-processing
# =============================================================================

@dataclass
class DPResult:
    thresholds: dict[Any, float]
    target_rate: float


def fit_dp(
    scores: np.ndarray,
    a: np.ndarray,
    target_rate: float | None = None,
    y: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> DPResult:
    r"""Algorithm 2: Demographic-parity post-processing.

    Choose per-group thresholds so :math:`\rho_a = \mathrm{target\_rate}`
    for all groups. This is the Hardt-Price-Srebro post-processor
    restricted to threshold rules (no randomization). Fairlearn's
    ``ThresholdOptimizer(constraints='demographic_parity')`` produces
    equivalent thresholds (within numerical precision); we cross-check
    in the benchmark script.

    Parameters
    ----------
    scores, a : arrays
        Scores and protected attribute on the validation set.
    target_rate : float, optional
        Common selection rate. If ``None``, defaults to ``mean(y)``.
    y : array, optional
        Used only to compute the default target rate when
        ``target_rate is None``.
    sample_weight : array, optional
        Survey weights for the default-target computation.
    """
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    if target_rate is None:
        if y is None:
            raise ValueError("Must supply target_rate or y")
        y = np.asarray(y).astype(int)
        if sample_weight is None:
            target_rate = float(np.mean(y))
        else:
            sw = np.asarray(sample_weight, dtype=float)
            target_rate = float(np.sum(sw * y) / np.sum(sw))

    groups = list(np.unique(a))
    thresholds = {
        g: _quantile_threshold(scores[a == g], float(target_rate))
        for g in groups
    }
    return DPResult(thresholds=thresholds, target_rate=float(target_rate))


def predict_dp(
    scores: np.ndarray, a: np.ndarray, result: DPResult
) -> np.ndarray:
    """Apply fitted DP thresholds."""
    return apply_per_group_thresholds(scores, a, result.thresholds)


# =============================================================================
# Algorithm 4: Group-wise calibration (isotonic) + global threshold
# =============================================================================

@dataclass
class CalibrationResult:
    calibrators: dict[Any, Any]  # IsotonicRegression or None
    threshold: float


def fit_calibration(
    scores: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    min_group_size: int = 50,
) -> CalibrationResult:
    """Algorithm 4: group-wise isotonic calibration, then global threshold.

    Recalibrates scores within each group via isotonic regression so
    that within-group reliability curves are aligned, then applies a
    single global threshold (default 0.5). Provided as a baseline; not
    the paper's recommended method.
    """
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    y = np.asarray(y).astype(int)
    calibrators: dict[Any, Any] = {}
    for g in np.unique(a):
        mask = a == g
        if int(mask.sum()) < min_group_size:
            calibrators[g] = None
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(scores[mask], y[mask])
        calibrators[g] = ir
    return CalibrationResult(calibrators=calibrators, threshold=float(threshold))


def predict_calibration(
    scores: np.ndarray, a: np.ndarray, result: CalibrationResult
) -> np.ndarray:
    """Apply fitted per-group calibrators, then threshold."""
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    y_hat = np.zeros_like(scores, dtype=int)
    for g, ir in result.calibrators.items():
        mask = a == g
        if not mask.any():
            continue
        if ir is None:
            y_hat[mask] = (scores[mask] >= result.threshold).astype(int)
        else:
            cal = ir.predict(scores[mask])
            y_hat[mask] = (cal >= result.threshold).astype(int)
    return y_hat


# =============================================================================
# Unmitigated baseline (single global threshold)
# =============================================================================

@dataclass
class UnmitResult:
    threshold: float
    target_rate: float = float("nan")


def fit_unmitigated(
    scores: np.ndarray,
    y: np.ndarray,
    target_rate: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> UnmitResult:
    r"""Fit a single global threshold so :math:`P(\hat Y = 1) \approx
    \mathrm{target\_rate}`.

    Default target rate is ``mean(y)`` (or weighted mean when survey
    weights are provided).
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    if target_rate is None:
        if sample_weight is None:
            target_rate = float(np.mean(y))
        else:
            sw = np.asarray(sample_weight, dtype=float)
            target_rate = float(np.sum(sw * y) / np.sum(sw))
    t = _quantile_threshold(scores, float(target_rate))
    return UnmitResult(threshold=t, target_rate=float(target_rate))


def predict_unmitigated(
    scores: np.ndarray, result: UnmitResult
) -> np.ndarray:
    return (np.asarray(scores) >= result.threshold).astype(int)
