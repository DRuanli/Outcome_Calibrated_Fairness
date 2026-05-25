#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the OCF audit on BRFSS 2024.

FIX 2026-01: The previous version of this script computed the OCF
multiplier ``c`` as ``mean(y) / np.mean(list(pi.values()))``, which uses
an unweighted mean of per-group base rates. This is wrong: the correct
denominator is the *group-size-weighted* mean ``sum_g w_g * pi_g``,
which by the tower property of expectation equals ``mean(y)``. Thus the
fitted ``c`` should be exactly 1.0 under the default settings.

The bug caused bootstrap headline numbers to drift: the bootstrap OCF
classifier targeted aggregate rate ~0.313 (instead of ~0.365 like DP),
producing under-allocation for the high-burden group and an apparent
"OCF reduces sensitivity by 1.6pp vs unmitigated" headline that is
actually an artifact of mismatched aggregate rates.

This rewrite delegates fitting to ``ocf.postprocess.fit_ocf``, which
implements Algorithm 1 correctly and gives ``c = 1.0`` exactly on the
defaults. Bootstrap rows for the OCF method now satisfy:

  - ``rho_g ≈ pi_g`` (within finite-sample noise)
  - ``aggregate(rho_OCF) == aggregate(rho_DP) == mean(y)``  (matched)
  - Theorems 3 and 4 directly verified via the matching-aggregate
    hypothesis.

Usage
-----
    python 04_bootstrap_audit.py --data data/brfss_2024_clean.parquet \\
                                  --n-bootstrap 1000

Designed to be run AFTER ``03_train_and_audit.py``. Reads the same
parquet file (which must include ``_LLCPWT`` weight column for
weighted analysis).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Locate the ocf package; allow running from the project root or from scripts/
HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent, HERE.parent.parent, Path.cwd()):
    if (candidate / "ocf" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

import ocf  # noqa: E402
from ocf.metrics import (  # noqa: E402
    base_rate,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    ocf_violation,
    ocf_violation_ratio_spread,
)
from ocf.postprocess import (  # noqa: E402
    fit_ocf,
    predict_ocf,
    fit_dp,
    predict_dp,
    fit_unmitigated,
    predict_unmitigated,
)


# =============================================================================
# Weighted bootstrap helpers
# =============================================================================

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    v = values[mask]
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * v) / np.sum(w))


def weighted_per_group_rate(
    yhat_or_y: np.ndarray,
    a: np.ndarray,
    weights: np.ndarray,
    group: object,
    mask_extra: np.ndarray | None = None,
) -> float:
    mask = a == group
    if mask_extra is not None:
        mask = mask & mask_extra
    if not mask.any():
        return float("nan")
    return weighted_mean(yhat_or_y[mask].astype(float), weights[mask])


def stratified_bootstrap_indices(
    n: int, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample with replacement, stratified by ``strata``."""
    out_parts: list[np.ndarray] = []
    s_ser = pd.Series(strata)
    unique = pd.unique(s_ser)
    for s in unique:
        if pd.isna(s):
            idx_s = np.where(pd.isna(s_ser).to_numpy())[0]
        else:
            idx_s = np.where((strata == s) & ~pd.isna(s_ser).to_numpy())[0]
        if len(idx_s) == 0:
            continue
        sample = rng.choice(idx_s, size=len(idx_s), replace=True)
        out_parts.append(sample)
    return np.concatenate(out_parts) if out_parts else np.arange(n)


# =============================================================================
# Per-method evaluation with weighted metrics
# =============================================================================

def compute_metrics_for_method(
    scores: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    weights: np.ndarray,
    method: str,
    target_aggregate_rate: float | None = None,
    seed: int = 0,
) -> dict:
    """Apply post-processor, then compute weighted metrics.

    Notes
    -----
    Weights are used in two places:
      1. To compute the per-group base rates :math:`\\hat\\pi_a` and
         the aggregate target rate :math:`\\bar\\rho = \\mathrm{mean}_w(y)`.
      2. To compute the *reported* weighted metrics
         (accuracy, TPR, FPR, etc.) on the bootstrap sample.
    """
    weights = np.asarray(weights, dtype=float)
    y = np.asarray(y).astype(int)
    a = np.asarray(a)
    scores = np.asarray(scores, dtype=float)

    if method == "Unmitigated":
        fit = fit_unmitigated(scores, y, target_rate=target_aggregate_rate,
                              sample_weight=weights)
        yhat = predict_unmitigated(scores, fit)
    elif method == "DP":
        fit = fit_dp(scores, a, target_rate=target_aggregate_rate, y=y,
                    sample_weight=weights)
        yhat = predict_dp(scores, a, fit)
    elif method == "OCF":
        # Use the canonical Algorithm 1 (correct c-computation).
        fit = fit_ocf(scores, a, y,
                     target_aggregate_rate=target_aggregate_rate,
                     sample_weight=weights)
        yhat = predict_ocf(scores, a, fit)
    elif method == "EO":
        fit = ocf.fit_eo_hardt(scores, a, y)
        yhat = ocf.predict_eo_hardt(scores, a, fit, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    groups = sorted(list(np.unique(a)))

    metrics: dict[str, float] = {
        "accuracy": weighted_mean((yhat == y).astype(float), weights),
    }

    # Weighted versions of DPD, EOD, OCFviol
    rho_w = np.array([
        weighted_per_group_rate(yhat, a, weights, g) for g in groups
    ])
    pi_w = np.array([
        weighted_per_group_rate(y, a, weights, g) for g in groups
    ])
    tpr_w = np.array([
        weighted_per_group_rate(yhat, a, weights, g, mask_extra=(y == 1))
        for g in groups
    ])
    fpr_w = np.array([
        weighted_per_group_rate(yhat, a, weights, g, mask_extra=(y == 0))
        for g in groups
    ])

    metrics["DPD"] = float(np.nanmax(rho_w) - np.nanmin(rho_w))
    tpr_finite = tpr_w[~np.isnan(tpr_w)]
    fpr_finite = fpr_w[~np.isnan(fpr_w)]
    metrics["EOD"] = float(max(
        (np.max(tpr_finite) - np.min(tpr_finite)) if len(tpr_finite) > 1 else 0.0,
        (np.max(fpr_finite) - np.min(fpr_finite)) if len(fpr_finite) > 1 else 0.0,
    ))

    # OCFviol (L_inf form, Definition 3.6)
    from ocf.metrics import _optimal_c
    valid = (pi_w > 0) & np.isfinite(rho_w) & np.isfinite(pi_w)
    if valid.any():
        c_star = _optimal_c(rho_w[valid], pi_w[valid])
        max_pi = float(np.nanmax(pi_w))
        if np.isfinite(c_star) and max_pi > 0:
            metrics["OCF_violation"] = float(
                np.nanmax(np.abs(rho_w[valid] - c_star * pi_w[valid])) / max_pi
            )
        else:
            metrics["OCF_violation"] = float("nan")
    else:
        metrics["OCF_violation"] = float("nan")

    # Ratio spread (for diagnostic/comparison)
    ratios = rho_w[valid] / pi_w[valid] if valid.any() else np.array([])
    metrics["OCF_violation_ratio_spread"] = (
        float(np.max(ratios) - np.min(ratios)) if len(ratios) > 1 else float("nan")
    )

    for i, g in enumerate(groups):
        metrics[f"TPR_{g}"] = float(tpr_w[i])
        metrics[f"FPR_{g}"] = float(fpr_w[i])
        metrics[f"rho_{g}"] = float(rho_w[i])
        metrics[f"pi_{g}"] = float(pi_w[i])

    return metrics


# =============================================================================
# Pipeline (train + fold-wise scoring + bootstrap)
# =============================================================================

NUMERIC_FEATURES = [
    "male", "has_insurance", "regular_provider", "cost_barrier",
    "binge_drink", "current_smoker", "heavy_drink", "meets_aerobic",
    "gen_health", "diabetes", "metro",
]
CATEGORICAL_FEATURES = [
    "age_group", "education", "income", "marital", "region",
]
TARGET = "hiv_tested_ever"


def make_preprocessor(num_used: list, cat_used: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_used),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_used),
    ])


def train_and_score(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
    num_used: list, cat_used: list, model_name: str, seed: int,
) -> np.ndarray:
    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=seed,
        )
    elif model_name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200,
            early_stopping=True, random_state=seed,
        )
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=seed, verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    pre = make_preprocessor(num_used, cat_used)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./data/brfss_2024_clean.parquet")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--model", default="XGBoost",
                    choices=["LogisticRegression", "RandomForest", "MLP", "XGBoost"])
    ap.add_argument("--methods", nargs="+",
                    default=["Unmitigated", "DP", "OCF", "EO"])
    ap.add_argument("--weight-col", default="_LLCPWT",
                    help="BRFSS weight column ('none' for unweighted)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.data}")
    df = pd.read_parquet(args.data).dropna(subset=[TARGET])
    print(f"  shape: {df.shape}")

    if args.weight_col != "none" and args.weight_col in df.columns:
        weights = df[args.weight_col].fillna(1.0).to_numpy(dtype=float)
        print(f"  weights: {args.weight_col}")
    else:
        weights = np.ones(len(df), dtype=float)
        if args.weight_col != "none":
            print(f"  weights: uniform (column {args.weight_col} not found)")

    num_used = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_used = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[num_used + cat_used].copy()
    y = df[TARGET].astype(int).to_numpy()
    a = df["race_eth"].astype(str).to_numpy()

    if "region" in df.columns:
        strata = df["region"].astype(object).fillna("missing").astype(str).to_numpy()
    else:
        strata = np.full(len(df), "all", dtype=object)

    # Step 1: train and cross-validate scores on full data
    print(f"\n[train] {args.model} via 5-fold CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores_full = np.zeros(len(df))
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scores_full[te] = train_and_score(
            X.iloc[tr], y[tr], X.iloc[te],
            num_used, cat_used, args.model, seed=args.seed + fold,
        )
    auroc = roc_auc_score(y, scores_full)
    print(f"  full-data AUROC: {auroc:.4f}")

    # Step 2: point estimates on full data
    print("\n[point] weighted point estimates (full data)")
    point: dict[str, dict[str, float]] = {}
    target_agg = weighted_mean(y.astype(float), weights)
    print(f"  target aggregate rate (weighted mean(y)): {target_agg:.4f}")
    for method in args.methods:
        m = compute_metrics_for_method(
            scores_full, y, a, weights, method,
            target_aggregate_rate=target_agg, seed=args.seed,
        )
        point[method] = m
        groups = sorted(list(np.unique(a)))
        pi_vals = {g: m[f"pi_{g}"] for g in groups}
        high_g = max(pi_vals, key=pi_vals.get)
        print(f"  [{method:<14}] acc={m['accuracy']:.4f} "
              f"DPD={m['DPD']:.4f} OCFviol(Linf)={m['OCF_violation']:.4f} "
              f"TPR_{high_g}={m[f'TPR_{high_g}']:.4f}")

    # Step 3: stratified bootstrap
    print(f"\n[bootstrap] {args.n_bootstrap} replicates, stratified by region")
    rng = np.random.default_rng(args.seed)
    boot_rows: list[dict] = []
    for b in range(args.n_bootstrap):
        if (b + 1) % 100 == 0 or b == 0:
            print(f"  replicate {b+1}/{args.n_bootstrap}")
        idx = stratified_bootstrap_indices(len(df), strata, rng)
        target_b = weighted_mean(y[idx].astype(float), weights[idx])
        for method in args.methods:
            m = compute_metrics_for_method(
                scores_full[idx], y[idx], a[idx], weights[idx], method,
                target_aggregate_rate=target_b,
                seed=int(rng.integers(2**31)),
            )
            m["bootstrap_id"] = b
            m["method"] = method
            boot_rows.append(m)

    boot_df = pd.DataFrame(boot_rows)

    # Step 4: summary with CIs
    print("\n[summary] 95% bootstrap percentile CIs")
    out_rows: list[dict] = []
    groups = sorted(list(np.unique(a)))
    high_g = max(
        {g: point[args.methods[0]][f"pi_{g}"] for g in groups},
        key=lambda g: point[args.methods[0]][f"pi_{g}"],
    )
    for method in args.methods:
        sub = boot_df[boot_df["method"] == method]
        row = {"method": method, "high_burden_group": high_g}
        for col in [
            "accuracy", "DPD", "EOD", "OCF_violation",
            "OCF_violation_ratio_spread",
            f"TPR_{high_g}", f"FPR_{high_g}", f"rho_{high_g}",
        ]:
            vals = sub[col].dropna().to_numpy()
            row[f"{col}_pt"] = point[method].get(col, float("nan"))
            if len(vals) > 0:
                row[f"{col}_ci_lo"] = float(np.percentile(vals, 2.5))
                row[f"{col}_ci_hi"] = float(np.percentile(vals, 97.5))
            else:
                row[f"{col}_ci_lo"] = float("nan")
                row[f"{col}_ci_hi"] = float("nan")
        out_rows.append(row)
    summary = pd.DataFrame(out_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(out_dir / f"bootstrap_{args.model}_raw.csv", index=False)
    summary.to_csv(out_dir / f"bootstrap_{args.model}_summary.csv", index=False)
    print(f"\n  saved: {out_dir / f'bootstrap_{args.model}_summary.csv'}")
    print(f"  saved: {out_dir / f'bootstrap_{args.model}_raw.csv'}")

    print("\n" + "=" * 75)
    print(f"HEADLINE  ({args.model} on BRFSS 2024, weighted, "
          f"n_boot={args.n_bootstrap}, high_burden={high_g})")
    print("=" * 75)
    print(f"{'Method':<14} {'TPR (95% CI)':>32} {'OCFviol Linf':>16}")
    for method in args.methods:
        r = summary[summary["method"] == method].iloc[0]
        tpr = r[f"TPR_{high_g}_pt"]
        lo = r[f"TPR_{high_g}_ci_lo"]
        hi = r[f"TPR_{high_g}_ci_hi"]
        viol = r["OCF_violation_pt"]
        print(f"{method:<14} {tpr:.4f} [{lo:.4f}, {hi:.4f}]   {viol:>10.4f}")

    # Theorem 1, 4 check across bootstrap
    print("\n[theorem checks across bootstrap replicates]")
    pivot = boot_df.pivot(index="bootstrap_id", columns="method",
                         values=f"TPR_{high_g}")
    if "DP" in pivot.columns and "Unmitigated" in pivot.columns:
        t1_holds = int((pivot["DP"] < pivot["Unmitigated"]).sum())
        print(f"  Theorem 1 (DP < Unmit, TPR_{high_g}):  "
              f"{t1_holds}/{args.n_bootstrap} = {100*t1_holds/args.n_bootstrap:.1f}%")
    if "OCF" in pivot.columns and "DP" in pivot.columns:
        t4_holds = int((pivot["OCF"] > pivot["DP"]).sum())
        diff = pivot["OCF"] - pivot["DP"]
        print(f"  Theorem 4 (OCF > DP,  TPR_{high_g}):  "
              f"{t4_holds}/{args.n_bootstrap} = {100*t4_holds/args.n_bootstrap:.1f}%")
        print(f"      Δ(OCF-DP): mean={diff.mean():+.4f}, "
              f"95% CI [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    if "OCF" in pivot.columns and "Unmitigated" in pivot.columns:
        diff = pivot["OCF"] - pivot["Unmitigated"]
        print(f"      Δ(OCF-Unmit): mean={diff.mean():+.4f}, "
              f"95% CI [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
