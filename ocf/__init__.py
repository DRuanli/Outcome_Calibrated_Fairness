"""
ocf: Outcome-Calibrated Fairness for binary classification.

Reference implementation of the post-processing algorithms and
fairness metrics from:

    [Authors] (2026). Outcome-Calibrated Fairness for HIV Risk
    Prediction in High-Disparity Populations.

Quick start
-----------
>>> import numpy as np
>>> from ocf import fit_ocf, predict_ocf, ocf_violation
>>> # scores: predicted P(Y=1 | X) from any classifier
>>> # a: protected attribute (categorical)
>>> # y: binary labels
>>> result = fit_ocf(scores, a, y)
>>> y_hat = predict_ocf(scores, a, result)
>>> viol = ocf_violation(y_hat, y, a)            # L_inf form (Def 3.6)

Module layout
-------------
- ``ocf.metrics``        : group-wise rates, DPD, EOD, OCF violation
- ``ocf.postprocess``    : Algorithm 1 (OCF), DP, EO, calibration, unmitigated
- ``ocf.eo_hardt``       : reference Hardt-Price-Srebro EO (LP-based, randomized)
- ``ocf.sensitivity``    : screening-multiplier sweeps for the c parameter
- ``ocf.evaluation``     : multi-method comparison harness with stratified bootstrap
- ``ocf.theory``         : helpers for verifying MLR (A2') and Bayes-optimal rates

All numeric routines vectorize over arrays. Group-by-group operations use
``np.unique(a)`` for ordering; pass ``protected_groups`` explicitly to control
the order in reported metrics.
"""
from __future__ import annotations

__version__ = "1.0.0"

# Re-export the most commonly used API at package level
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
    calibration_disparity,
)
from ocf.postprocess import (
    fit_ocf,
    predict_ocf,
    OCFResult,
    fit_dp,
    predict_dp,
    DPResult,
    fit_calibration,
    predict_calibration,
    CalibrationResult,
    fit_unmitigated,
    predict_unmitigated,
    UnmitResult,
)
from ocf.eo_hardt import (
    fit_eo_hardt,
    predict_eo_hardt,
    EOHardtResult,
)
from ocf.evaluation import evaluate_all

__all__ = [
    "__version__",
    # metrics
    "base_rate",
    "selection_rate",
    "true_positive_rate",
    "false_positive_rate",
    "positive_predictive_value",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "ocf_violation",
    "ocf_violation_ratio_spread",
    "calibration_disparity",
    # post-processors
    "fit_ocf",
    "predict_ocf",
    "OCFResult",
    "fit_dp",
    "predict_dp",
    "DPResult",
    "fit_eo_hardt",
    "predict_eo_hardt",
    "EOHardtResult",
    "fit_calibration",
    "predict_calibration",
    "CalibrationResult",
    "fit_unmitigated",
    "predict_unmitigated",
    "UnmitResult",
    # harness
    "evaluate_all",
]
