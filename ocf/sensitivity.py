"""
ocf.sensitivity — sensitivity analysis for the screening multiplier c.

Provides utilities to evaluate OCF across a sweep of values of the
screening multiplier ``c``, returning per-c per-group selection rates,
TPRs, and accuracy. Used to produce the sensitivity figure in the
paper's appendix and to answer Reviewer-friendly questions of the form
"how robust are the OCF claims to the choice of c?".
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ocf.metrics import (
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    positive_predictive_value,
    ocf_violation,
    demographic_parity_difference,
)
from ocf.postprocess import fit_ocf, predict_ocf


def sensitivity_sweep(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    c_values: Sequence[float] | None = None,
    high_burden_group: Any | None = None,
    protected_groups: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    r"""Sweep OCF over a range of screening multipliers ``c``.

    For each ``c`` in ``c_values``, fit OCF with that fixed multiplier,
    then evaluate every group-wise metric.

    Parameters
    ----------
    scores, y, a : arrays
        Validation set scores, labels, protected attribute.
    c_values : sequence of float, optional
        Multipliers to evaluate. Default: ``np.linspace(0.5, 1.5, 21)``,
        covering 50%-150% of base-rate-matched allocation.
    high_burden_group : value, optional
        If specified, returns shortcut columns ``TPR_high_burden`` and
        ``rho_high_burden`` for that group.
    protected_groups : sequence, optional
        Group ordering.

    Returns
    -------
    list of dict
        Each dict has ``c``, ``accuracy``, ``DPD``, ``OCF_violation``,
        per-group ``TPR_{g}``, ``rho_{g}``, ``FPR_{g}``, ``PPV_{g}``,
        and (if ``high_burden_group`` given) ``TPR_high_burden`` and
        ``rho_high_burden``.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    a = np.asarray(a)
    if c_values is None:
        c_values = np.linspace(0.5, 1.5, 21)
    if protected_groups is None:
        groups = list(np.unique(a))
    else:
        groups = list(protected_groups)

    rows: list[dict[str, Any]] = []
    for c in c_values:
        fit = fit_ocf(scores, a, y, c=float(c))
        y_hat = predict_ocf(scores, a, fit)
        row: dict[str, Any] = {
            "c": float(c),
            "accuracy": float(np.mean(y_hat == y)),
            "DPD": demographic_parity_difference(y_hat, a, groups=groups),
            "OCF_violation": ocf_violation(y_hat, y, a, groups=groups),
        }
        for g in groups:
            row[f"rho_{g}"] = selection_rate(y_hat, a, g)
            row[f"TPR_{g}"] = true_positive_rate(y_hat, y, a, g)
            row[f"FPR_{g}"] = false_positive_rate(y_hat, y, a, g)
            row[f"PPV_{g}"] = positive_predictive_value(y_hat, y, a, g)
        if high_burden_group is not None:
            row["TPR_high_burden"] = true_positive_rate(
                y_hat, y, a, high_burden_group
            )
            row["rho_high_burden"] = selection_rate(
                y_hat, a, high_burden_group
            )
        rows.append(row)
    return rows


def find_c_matching_metric(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    target_metric: str,
    target_value: float,
    c_range: tuple[float, float] = (0.1, 3.0),
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Find the OCF multiplier ``c`` that yields the specified metric
    value via bisection search.

    Parameters
    ----------
    target_metric : str
        One of ``'aggregate_selection_rate'``, ``'high_burden_tpr'``.
        Both are monotone in ``c`` so bisection is valid.
    target_value : float
        Target value for the chosen metric.
    c_range : (float, float)
        Search bracket.
    tol : float
        Absolute tolerance on ``c``.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    c : float
        The screening multiplier achieving the target.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    a = np.asarray(a)
    pi_max_group = max(np.unique(a), key=lambda g: float(np.mean(y[a == g])))

    def metric_at_c(c: float) -> float:
        fit = fit_ocf(scores, a, y, c=c)
        y_hat = predict_ocf(scores, a, fit)
        if target_metric == "aggregate_selection_rate":
            return float(np.mean(y_hat))
        elif target_metric == "high_burden_tpr":
            return true_positive_rate(y_hat, y, a, pi_max_group)
        else:
            raise ValueError(f"Unknown target_metric: {target_metric}")

    lo, hi = c_range
    f_lo, f_hi = metric_at_c(lo), metric_at_c(hi)
    if (f_lo - target_value) * (f_hi - target_value) > 0:
        # No sign change in the bracket; return the closer endpoint
        if abs(f_lo - target_value) < abs(f_hi - target_value):
            return float(lo)
        return float(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = metric_at_c(mid)
        if abs(f_mid - target_value) < tol or (hi - lo) < tol:
            return float(mid)
        if (f_lo - target_value) * (f_mid - target_value) <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))
