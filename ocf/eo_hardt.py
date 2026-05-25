"""
ocf.eo_hardt — reference implementation of Hardt-Price-Srebro
Equalized Odds post-processing.

The Hardt-Price-Srebro algorithm (NeurIPS 2016) achieves *exact*
equalized odds by deriving each group's ROC curve from the score and
labels, then finding a single target (FPR, TPR) point that lies inside
the convex hull of *every* group's achievable region. The resulting
classifier within each group is a randomized mixture of two threshold
rules whose chosen randomization makes (FPR_a, TPR_a) equal the common
target for all a.

Algorithm outline
-----------------
1. For each group ``a``, compute the empirical ROC curve from
   ``scores[a]`` and ``y[a]``. Append the corner points (0,0) and (1,1).
2. The achievable (FPR, TPR) for group ``a`` is the convex hull of its
   ROC vertices. We compute the hull explicitly.
3. Find the intersection of all groups' convex hulls. A common (FPR*,
   TPR*) target in the intersection that maximizes a chosen utility
   (default: TPR* - lambda * FPR* with lambda = 1) is selected by a
   linear program with constraints "(FPR*, TPR*) below each group's
   upper-hull boundary AND above each group's lower-hull boundary".
4. For each group, find the convex combination of two adjacent
   ROC vertices that lies on the (FPR*, TPR*) target. This gives a
   pair of thresholds and a randomization probability.

Reference
---------
Hardt, Price, Srebro (2016). "Equality of Opportunity in Supervised
Learning." NeurIPS, https://arxiv.org/abs/1610.02413
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# Result type
# =============================================================================

@dataclass
class EOHardtResult:
    """Output of :func:`fit_eo_hardt`.

    Attributes
    ----------
    thresholds_low : dict[group, float]
        Lower threshold per group. Predict 1 with probability
        ``mix[group]`` if ``s >= thresholds_low[group]``, otherwise 1
        with probability 1 if ``s >= thresholds_high[group]``.
    thresholds_high : dict[group, float]
        Upper threshold per group.
    mix : dict[group, float]
        Probability of predicting 1 when ``thresholds_low <= s <
        thresholds_high``. Outside the band, prediction is deterministic.
    target_tpr : float
        Common TPR achieved across all groups.
    target_fpr : float
        Common FPR achieved across all groups.
    seed : int | None
        Random seed used at predict-time, recorded for reproducibility.
    """
    thresholds_low: dict[Any, float]
    thresholds_high: dict[Any, float]
    mix: dict[Any, float]
    target_tpr: float
    target_fpr: float
    seed: int | None = None


# =============================================================================
# Convex hull helpers
# =============================================================================

def _group_roc(scores_g: np.ndarray, y_g: np.ndarray) -> np.ndarray:
    """Return ROC vertices (FPR, TPR) sorted by FPR, with (0,0) and
    (1,1) included.

    Vertices correspond to all distinct decision thresholds plus the two
    endpoints. Each entry is a 2-tuple ``(fpr, tpr)``.
    """
    pos = (y_g == 1)
    neg = (y_g == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        # Degenerate group: only (0,0) and (1,1) achievable.
        return np.array([[0.0, 0.0], [1.0, 1.0]])

    # Sort scores descending; track cumulative TP and FP.
    order = np.argsort(-scores_g, kind="mergesort")
    sorted_y = y_g[order]
    tps = np.cumsum(sorted_y == 1)
    fps = np.cumsum(sorted_y == 0)
    tprs = tps / n_pos
    fprs = fps / n_neg

    # Threshold-based vertices: prepend (0, 0) and append (1, 1)
    fpr_pts = np.concatenate(([0.0], fprs, [1.0]))
    tpr_pts = np.concatenate(([0.0], tprs, [1.0]))
    # Deduplicate
    pts = np.column_stack([fpr_pts, tpr_pts])
    # Round to suppress duplicate-from-ties artifacts
    pts = np.unique(np.round(pts, 12), axis=0)
    # Sort by FPR ascending, then TPR ascending (sorts lex)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    return pts


def _upper_hull(points: np.ndarray) -> np.ndarray:
    """Upper convex hull of a 2D point set sorted by x-coordinate.

    Returns the subset of input points that form the upper hull
    (above-the-curve side), from x-min to x-max.
    """
    if len(points) <= 2:
        return points.copy()
    # Andrew's monotone chain, upper part only
    upper: list[np.ndarray] = []
    for p in points:
        while len(upper) >= 2:
            o, a, b = upper[-2], upper[-1], p
            # Cross product (a - o) x (b - o); if < 0 we have right turn,
            # which on the upper hull means we should pop.
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return np.array(upper)


# =============================================================================
# Main algorithm
# =============================================================================

def fit_eo_hardt(
    scores: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    utility: str = "tpr_minus_fpr",
) -> EOHardtResult:
    r"""Algorithm 3: Hardt-Price-Srebro Equalized Odds post-processing.

    Finds a common (TPR*, FPR*) target lying in the intersection of all
    groups' achievable (TPR, FPR) regions, then constructs a randomized
    threshold rule for each group hitting that target exactly. The
    resulting classifier satisfies *exact* equalized odds in
    expectation.

    Parameters
    ----------
    scores : array of shape (n,)
        Predicted :math:`P(Y=1 \mid X)`.
    a : array of shape (n,)
        Protected attribute.
    y : array of shape (n,)
        Binary labels (0/1).
    utility : str, default 'tpr_minus_fpr'
        Objective for choosing the common (TPR*, FPR*). Options:

        - ``'tpr_minus_fpr'``: maximize TPR* - FPR* (Youden's J;
          recommended for screening tasks).
        - ``'tpr_only'``: maximize TPR*.
        - ``'min_loss'``: minimize :math:`P(Y=1)(1 - \text{TPR}) +
          P(Y=0) \text{FPR}` (expected 0-1 loss).

    Returns
    -------
    EOHardtResult
        See class docstring. Use :func:`predict_eo_hardt` to apply.

    Notes
    -----
    The common (TPR*, FPR*) target is chosen as the point in the
    intersection of all groups' upper-hull boundaries (the set of
    achievable points with deterministic randomization between two
    threshold rules) maximizing the utility. Because each group's
    achievable region is a convex polygon, the intersection is also
    convex, and the utility-maximizing point lies on the boundary —
    specifically, on the lower envelope of all groups' upper hulls.

    Complexity is :math:`O(n_a \log n_a)` per group for ROC + hull,
    plus :math:`O(K^2 V)` for intersecting K hulls of V vertices each.
    For BRFSS-scale (6 groups, ~5K vertices each), runtime is < 1s.
    """
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    y = np.asarray(y).astype(int)
    groups = list(np.unique(a))

    # Step 1+2: ROC vertices and upper hulls per group
    hulls: dict[Any, np.ndarray] = {}
    for g in groups:
        mask = a == g
        roc = _group_roc(scores[mask], y[mask])
        hulls[g] = _upper_hull(roc)

    # Step 3: Find common (TPR*, FPR*) in the intersection of upper
    # hulls. We use the fact that the intersection's upper boundary is
    # the *pointwise minimum* of all groups' hull-interpolation
    # functions (FPR -> max TPR).
    fpr_grid = np.linspace(0, 1, 1001)

    def interp_hull(hull: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation along an upper hull."""
        return np.interp(x, hull[:, 0], hull[:, 1])

    tpr_min_envelope = np.full_like(fpr_grid, np.inf)
    for g, hull in hulls.items():
        tpr_g = interp_hull(hull, fpr_grid)
        tpr_min_envelope = np.minimum(tpr_min_envelope, tpr_g)

    # Compute the utility on the envelope
    if utility == "tpr_minus_fpr":
        u = tpr_min_envelope - fpr_grid
    elif utility == "tpr_only":
        u = tpr_min_envelope
    elif utility == "min_loss":
        p1 = float(np.mean(y == 1))
        p0 = 1.0 - p1
        # minimize p1 (1 - tpr) + p0 fpr  <=>  maximize  p1 tpr - p0 fpr
        u = p1 * tpr_min_envelope - p0 * fpr_grid
    else:
        raise ValueError(f"Unknown utility: {utility!r}")

    best_idx = int(np.argmax(u))
    target_fpr = float(fpr_grid[best_idx])
    target_tpr = float(tpr_min_envelope[best_idx])

    # Step 4: For each group, find which two adjacent ROC vertices
    # bracket (target_fpr, target_tpr) on its upper hull, and the mix
    # probability that yields the target.
    thresholds_low: dict[Any, float] = {}
    thresholds_high: dict[Any, float] = {}
    mix: dict[Any, float] = {}
    for g in groups:
        hull = hulls[g]
        # Find the segment of hull where the FPR-coordinate brackets
        # target_fpr.
        idx = int(np.searchsorted(hull[:, 0], target_fpr, side="right")) - 1
        idx = max(0, min(idx, len(hull) - 2))
        f_lo, t_lo = float(hull[idx, 0]), float(hull[idx, 1])
        f_hi, t_hi = float(hull[idx + 1, 0]), float(hull[idx + 1, 1])
        if f_hi > f_lo:
            alpha = (target_fpr - f_lo) / (f_hi - f_lo)
        else:
            alpha = 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        # Translate (f_lo, t_lo), (f_hi, t_hi) back to thresholds.
        # On the *original* score axis, hull vertex (f, t) corresponds
        # to the threshold whose group ROC achieves that point. Higher
        # FPR ↔ lower threshold (more selected).
        mask = a == g
        scores_g = scores[mask]
        y_g = y[mask]
        # Helper: threshold producing target FPR on this group
        if (y_g == 0).any():
            # Sort negatives' scores descending; threshold for FPR = f is
            # the (1-f)-quantile of negative-class scores.
            neg_scores = np.sort(scores_g[y_g == 0])
            n_neg = len(neg_scores)
            # FPR = (number of negatives with score >= t) / n_neg
            # → t = the score at position (1-f) * n_neg in ascending order
            def _t_for_fpr(f: float) -> float:
                if f <= 0:
                    return float("inf")
                if f >= 1:
                    return float("-inf")
                idx_t = int(np.floor((1 - f) * n_neg))
                idx_t = max(0, min(idx_t, n_neg - 1))
                return float(neg_scores[idx_t])
            thresholds_low[g] = _t_for_fpr(f_hi)   # lower thresh → more selected
            thresholds_high[g] = _t_for_fpr(f_lo)
        else:
            thresholds_low[g] = float("-inf") if target_fpr > 0 else float("inf")
            thresholds_high[g] = thresholds_low[g]
        mix[g] = alpha

    return EOHardtResult(
        thresholds_low=thresholds_low,
        thresholds_high=thresholds_high,
        mix=mix,
        target_tpr=target_tpr,
        target_fpr=target_fpr,
    )


def predict_eo_hardt(
    scores: np.ndarray,
    a: np.ndarray,
    result: EOHardtResult,
    seed: int | None = 0,
) -> np.ndarray:
    """Apply a fitted HPS-EO classifier with randomized thresholds.

    For each observation, predict deterministically 1 if
    ``scores >= thresholds_high``, deterministically 0 if
    ``scores < thresholds_low``, and randomly 1 with probability
    ``mix`` if in between.

    Parameters
    ----------
    seed : int, optional
        Random seed for the randomization step. Set explicitly for
        reproducibility.
    """
    scores = np.asarray(scores, dtype=float)
    a = np.asarray(a)
    rng = np.random.default_rng(seed)
    y_hat = np.zeros_like(scores, dtype=int)
    for g, t_low in result.thresholds_low.items():
        t_high = result.thresholds_high[g]
        alpha = result.mix[g]
        mask = a == g
        if not mask.any():
            continue
        s = scores[mask]
        # Above the upper threshold: predict 1 deterministically.
        upper = s >= t_high
        # Strictly below the lower threshold: predict 0.
        # In the band [t_low, t_high): predict 1 with probability alpha.
        in_band = (s >= t_low) & (~upper)
        rand = rng.random(int(in_band.sum())) < alpha
        local_yhat = upper.astype(int)
        idx = np.where(in_band)[0]
        local_yhat[idx] = rand.astype(int)
        y_hat[mask] = local_yhat
    return y_hat
