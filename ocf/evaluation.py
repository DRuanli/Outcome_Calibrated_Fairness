"""
ocf.evaluation — multi-method comparison harness.

Provides :func:`evaluate_all` which fits every available post-processing
method on a (scores, a, y) triple and returns per-group metrics in a
long-form ``list[dict]``. Used in the pipeline scripts to produce the
``all_metrics.csv``, ``headline_table.csv``, and bootstrap summary
files reported in the paper.

Also provides :func:`stratified_bootstrap_audit` which wraps
``evaluate_all`` with a stratified bootstrap to compute confidence
intervals.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

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
)
from ocf.postprocess import (
    fit_ocf,
    predict_ocf,
    fit_dp,
    predict_dp,
    fit_calibration,
    predict_calibration,
    fit_unmitigated,
    predict_unmitigated,
)
from ocf.eo_hardt import fit_eo_hardt, predict_eo_hardt


def _row_for_method(
    method: str,
    y_hat: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    groups: Sequence[Any],
) -> dict[str, Any]:
    """Compute all standard metrics for one (method, y_hat) pair."""
    row: dict[str, Any] = {"method": method}
    row["accuracy"] = float(np.mean(y_hat == y))
    row["DPD"] = demographic_parity_difference(y_hat, a, groups=groups)
    row["EOD"] = equalized_odds_difference(y_hat, y, a, groups=groups)
    row["OCF_violation"] = ocf_violation(y_hat, y, a, groups=groups)
    row["OCF_violation_ratio_spread"] = ocf_violation_ratio_spread(
        y_hat, y, a, groups=groups
    )
    for g in groups:
        row[f"rho_{g}"] = selection_rate(y_hat, a, g)
        row[f"TPR_{g}"] = true_positive_rate(y_hat, y, a, g)
        row[f"FPR_{g}"] = false_positive_rate(y_hat, y, a, g)
        row[f"PPV_{g}"] = positive_predictive_value(y_hat, y, a, g)
        row[f"pi_{g}"] = base_rate(y, a, g)
    return row


def evaluate_all(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    protected_groups: Sequence[Any] | None = None,
    *,
    seed: int = 0,
    methods: Sequence[str] | None = None,
    ocf_c: float | None = None,
    ocf_target_aggregate_rate: float | None = None,
) -> dict[str, Any]:
    """Fit every post-processor and return per-method metrics.

    Parameters
    ----------
    scores, y, a : arrays
        Validation-set scores, labels, and protected attribute.
    protected_groups : sequence, optional
        Ordered list of group labels (default: ``np.unique(a)``).
    seed : int
        Random seed for EO randomization.
    methods : sequence of str, optional
        Subset of ``{'Unmitigated', 'DP', 'EO', 'Calibration', 'OCF'}``
        to evaluate. Default is all five.
    ocf_c, ocf_target_aggregate_rate : optional
        Passed through to :func:`ocf.postprocess.fit_ocf`. Useful for
        sensitivity analysis over c.

    Returns
    -------
    dict with keys
        - ``rows``: list of per-method metric dicts
        - ``preds``: dict ``method -> y_hat array``
        - ``fits``: dict ``method -> fitted result dataclass``
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    a = np.asarray(a)

    if methods is None:
        methods = ("Unmitigated", "DP", "EO", "Calibration", "OCF")

    if protected_groups is None:
        groups = list(np.unique(a))
    else:
        groups = list(protected_groups)

    fits: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}

    if "Unmitigated" in methods:
        fits["Unmitigated"] = fit_unmitigated(scores, y)
        preds["Unmitigated"] = predict_unmitigated(scores, fits["Unmitigated"])
    if "DP" in methods:
        fits["DP"] = fit_dp(scores, a, y=y)
        preds["DP"] = predict_dp(scores, a, fits["DP"])
    if "EO" in methods:
        fits["EO"] = fit_eo_hardt(scores, a, y)
        preds["EO"] = predict_eo_hardt(scores, a, fits["EO"], seed=seed)
    if "Calibration" in methods:
        fits["Calibration"] = fit_calibration(scores, a, y, threshold=0.5)
        preds["Calibration"] = predict_calibration(scores, a, fits["Calibration"])
    if "OCF" in methods:
        fits["OCF"] = fit_ocf(
            scores,
            a,
            y,
            c=ocf_c,
            target_aggregate_rate=ocf_target_aggregate_rate,
        )
        preds["OCF"] = predict_ocf(scores, a, fits["OCF"])

    rows = [_row_for_method(m, preds[m], y, a, groups) for m in methods if m in preds]

    return {"rows": rows, "preds": preds, "fits": fits}


def stratified_bootstrap_audit(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    strata: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    methods: Sequence[str] | None = None,
    protected_groups: Sequence[Any] | None = None,
    seed: int = 42,
    ocf_c: float | None = None,
    ocf_target_aggregate_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Stratified bootstrap audit with replacement.

    For each replicate ``b = 0, ..., n_bootstrap - 1``, resample with
    replacement within ``strata``, then call :func:`evaluate_all` and
    record per-method metrics. The original (point) estimate uses the
    full data, returned as the row ``bootstrap_id = -1``.

    Parameters
    ----------
    strata : array, optional
        Stratification variable (e.g., census region). If ``None``, use
        a single stratum (i.i.d. bootstrap).
    n_bootstrap : int
        Number of bootstrap replicates.
    methods : sequence of str, optional
        See :func:`evaluate_all`.
    seed : int
        Base random seed; replicate ``b`` uses ``seed + b``.

    Returns
    -------
    list of dicts
        Each dict contains: ``bootstrap_id`` (-1 for point estimate),
        ``method``, and all metric columns.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    a = np.asarray(a)
    n = len(y)
    if strata is None:
        strata = np.zeros(n, dtype=int)
    else:
        strata = np.asarray(strata)

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, Any]] = []

    # Point estimate
    res = evaluate_all(
        scores,
        y,
        a,
        protected_groups=protected_groups,
        seed=int(rng.integers(2**31)),
        methods=methods,
        ocf_c=ocf_c,
        ocf_target_aggregate_rate=ocf_target_aggregate_rate,
    )
    for row in res["rows"]:
        row["bootstrap_id"] = -1
        out_rows.append(row)

    # Stratified bootstrap by region
    unique_strata, stratum_idx = np.unique(strata, return_inverse=True)
    strata_indices = [np.where(stratum_idx == k)[0] for k in range(len(unique_strata))]

    for b in range(n_bootstrap):
        boot_idx_parts = []
        for s_idx in strata_indices:
            if len(s_idx) == 0:
                continue
            sample = rng.choice(s_idx, size=len(s_idx), replace=True)
            boot_idx_parts.append(sample)
        boot_idx = np.concatenate(boot_idx_parts)

        res_b = evaluate_all(
            scores[boot_idx],
            y[boot_idx],
            a[boot_idx],
            protected_groups=protected_groups,
            seed=int(rng.integers(2**31)),
            methods=methods,
            ocf_c=ocf_c,
            ocf_target_aggregate_rate=ocf_target_aggregate_rate,
        )
        for row in res_b["rows"]:
            row["bootstrap_id"] = b
            out_rows.append(row)

    return out_rows
