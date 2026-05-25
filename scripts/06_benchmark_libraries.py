#!/usr/bin/env python3
"""
Benchmark OCF against Fairlearn, AIF360, and Aequitas.

This script directly fits and evaluates the same dataset under:

  - **OCF**          (our method, ``ocf.fit_ocf``)
  - **Fairlearn**    ``ThresholdOptimizer(constraints='demographic_parity')``
                     and ``constraints='equalized_odds'``,
                     ``ExponentiatedGradient(DemographicParity())``
  - **AIF360**       ``Reweighing`` preprocessor + LogisticRegression,
                     ``EqOddsPostprocessing``
  - **Aequitas**     audit report on baseline scores (Aequitas is an
                     auditing-only library; we report its disparity
                     metrics on the *same* unmitigated, DP-mitigated,
                     and OCF-mitigated predictions for cross-validation
                     of our DPD/EOD/OCF-violation computations).

Outputs ``benchmark_<MODEL>.csv`` with per-method metrics from each
library, suitable for the paper's "OCF vs industry tools" comparison
table (proposed as Appendix Table A4).

Dependencies
------------
- fairlearn >= 0.10
- aif360    >= 0.6  (also pulls in tensorflow as optional)
- aequitas  >= 1.0

Install with:
    pip install fairlearn aif360 aequitas

If any library is missing, the script skips its rows and prints a
warning. The OCF rows are always produced regardless.

Usage
-----
    python 06_benchmark_libraries.py --data data/brfss_2024_clean.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, HERE.parent.parent, Path.cwd()):
    if (cand / "ocf" / "__init__.py").exists():
        sys.path.insert(0, str(cand))
        break

import ocf  # noqa: E402

# Optional library imports — fail gracefully
try:
    import fairlearn  # noqa: F401
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import (
        ExponentiatedGradient,
        DemographicParity as FL_DemographicParity,
    )
    HAVE_FAIRLEARN = True
except ImportError as e:
    print(f"[fairlearn] not installed: {e}")
    HAVE_FAIRLEARN = False

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.postprocessing import EqOddsPostprocessing
    HAVE_AIF360 = True
except ImportError as e:
    print(f"[aif360] not installed: {e}")
    HAVE_AIF360 = False

try:
    from aequitas.group import Group
    from aequitas.bias import Bias
    HAVE_AEQUITAS = True
except ImportError as e:
    print(f"[aequitas] not installed: {e}")
    HAVE_AEQUITAS = False


NUMERIC = [
    "male", "has_insurance", "regular_provider", "cost_barrier",
    "binge_drink", "current_smoker", "heavy_drink", "meets_aerobic",
    "gen_health", "diabetes", "metro",
]
CATEGORICAL = ["age_group", "education", "income", "marital", "region"]
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


def train_and_score(X_tr, y_tr, X_te, num_used, cat_used, model_name, seed):
    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=seed,
        )
    elif model_name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                            early_stopping=True, random_state=seed)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=seed, verbosity=0)
    else:
        raise ValueError(model_name)
    pipe = Pipeline([("pre", make_preprocessor(num_used, cat_used)),
                     ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    return pipe, pipe.predict_proba(X_te)[:, 1]


def metrics_row(method: str, library: str, y_hat: np.ndarray,
                y: np.ndarray, a: np.ndarray, high_g: str) -> dict:
    """Compute standardized metrics for a given (library, method, y_hat)."""
    from ocf.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        ocf_violation,
        true_positive_rate,
        false_positive_rate,
        selection_rate,
        base_rate,
    )
    return {
        "library": library,
        "method": method,
        "accuracy": float(np.mean(y_hat == y)),
        "DPD": demographic_parity_difference(y_hat, a),
        "EOD": equalized_odds_difference(y_hat, y, a),
        "OCF_violation": ocf_violation(y_hat, y, a),
        f"TPR_{high_g}": true_positive_rate(y_hat, y, a, high_g),
        f"FPR_{high_g}": false_positive_rate(y_hat, y, a, high_g),
        f"rho_{high_g}": selection_rate(y_hat, a, high_g),
        f"pi_{high_g}": base_rate(y, a, high_g),
    }


# =============================================================================
# Per-library evaluation
# =============================================================================

def evaluate_ocf_ours(scores_tr, y_tr, a_tr, scores_te, y_te, a_te,
                      high_g) -> list[dict]:
    """Our OCF implementation, plus our DP and unmitigated baselines."""
    rows: list[dict] = []
    # OCF
    fit = ocf.fit_ocf(scores_tr, a_tr, y_tr)
    yhat = ocf.predict_ocf(scores_te, a_te, fit)
    rows.append(metrics_row("OCF", "ocf (ours)", yhat, y_te, a_te, high_g))
    # Our DP
    fit = ocf.fit_dp(scores_tr, a_tr, y=y_tr)
    yhat = ocf.predict_dp(scores_te, a_te, fit)
    rows.append(metrics_row("DP", "ocf (ours)", yhat, y_te, a_te, high_g))
    # Our EO
    fit = ocf.fit_eo_hardt(scores_tr, a_tr, y_tr)
    yhat = ocf.predict_eo_hardt(scores_te, a_te, fit, seed=0)
    rows.append(metrics_row("EO (HPS)", "ocf (ours)", yhat, y_te, a_te, high_g))
    # Unmitigated
    fit = ocf.fit_unmitigated(scores_tr, y_tr)
    yhat = ocf.predict_unmitigated(scores_te, fit)
    rows.append(metrics_row("Unmitigated", "ocf (ours)", yhat, y_te, a_te, high_g))
    return rows


def evaluate_fairlearn(
    scores_tr, y_tr, a_tr, scores_te, y_te, a_te,
    X_tr, X_te, pipe_trained, high_g,
) -> list[dict]:
    """Fairlearn ThresholdOptimizer for DP and EO, plus
    ExponentiatedGradient(DP)."""
    rows: list[dict] = []
    if not HAVE_FAIRLEARN:
        return rows

    # ThresholdOptimizer DP (post-processing). Fits on validation scores.
    # The 'estimator' arg is required but bypassed via prefit; we use a
    # custom callable to inject scores instead of refitting.
    try:
        to_dp = ThresholdOptimizer(
            estimator=pipe_trained,
            constraints="demographic_parity",
            prefit=True,
            predict_method="predict_proba",
        )
        to_dp.fit(X_tr, y_tr, sensitive_features=a_tr)
        yhat = to_dp.predict(X_te, sensitive_features=a_te).astype(int)
        rows.append(metrics_row("DP (ThresholdOpt)", "fairlearn",
                                yhat, y_te, a_te, high_g))
    except Exception as e:
        print(f"  [fairlearn DP] failed: {e}")

    # ThresholdOptimizer EO
    try:
        to_eo = ThresholdOptimizer(
            estimator=pipe_trained,
            constraints="equalized_odds",
            prefit=True,
            predict_method="predict_proba",
        )
        to_eo.fit(X_tr, y_tr, sensitive_features=a_tr)
        yhat = to_eo.predict(X_te, sensitive_features=a_te).astype(int)
        rows.append(metrics_row("EO (ThresholdOpt)", "fairlearn",
                                yhat, y_te, a_te, high_g))
    except Exception as e:
        print(f"  [fairlearn EO] failed: {e}")

    # ExponentiatedGradient (in-processing); use a simple LR base estimator
    # to keep runtime manageable. ExpGrad fits a full mitigated model.
    try:
        base = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        # Build a one-hot-encoded design matrix for ExpGrad (it needs
        # a sklearn estimator, not a Pipeline that includes column
        # transformers + sensitive features in one shot).
        pre = make_preprocessor(
            [c for c in NUMERIC if c in X_tr.columns],
            [c for c in CATEGORICAL if c in X_tr.columns],
        )
        Xtr_mat = pre.fit_transform(X_tr)
        Xte_mat = pre.transform(X_te)
        eg = ExponentiatedGradient(base, constraints=FL_DemographicParity(),
                                    max_iter=20, eps=0.01)
        eg.fit(Xtr_mat, y_tr, sensitive_features=a_tr)
        yhat = eg.predict(Xte_mat).astype(int)
        rows.append(metrics_row("ExpGrad (DP)", "fairlearn",
                                yhat, y_te, a_te, high_g))
    except Exception as e:
        print(f"  [fairlearn ExpGrad] failed: {e}")
    return rows


def evaluate_aif360(
    X_tr, y_tr, a_tr, X_te, y_te, a_te,
    num_used, cat_used, high_g,
) -> list[dict]:
    """AIF360 Reweighing (pre-processing) + LogisticRegression and
    EqOddsPostprocessing."""
    rows: list[dict] = []
    if not HAVE_AIF360:
        return rows

    # AIF360 needs a BinaryLabelDataset: pandas DataFrame with the
    # protected attribute and label as separate columns. We encode
    # race_eth as integers 0..K-1.
    groups = sorted(list(set(a_tr) | set(a_te)))
    a_map = {g: i for i, g in enumerate(groups)}
    a_tr_int = np.array([a_map[g] for g in a_tr])
    a_te_int = np.array([a_map[g] for g in a_te])

    # For AIF360 we need to bring features into a numeric design matrix.
    # Use our preprocessor without scaling for the test of Reweighing.
    pre = make_preprocessor(num_used, cat_used)
    X_tr_mat = pre.fit_transform(X_tr)
    X_te_mat = pre.transform(X_te)

    # Reweighing requires "privileged_groups" and "unprivileged_groups"
    # specified as a list of dicts. Use the highest-burden group as
    # unprivileged in the sense "we want to make sure it isn't
    # disadvantaged".
    pi_per_group = {g: float(np.mean(y_tr[a_tr == g])) for g in groups}
    high_int = a_map[high_g]
    low_g = min(pi_per_group, key=pi_per_group.get)
    low_int = a_map[low_g]
    privileged = [{"race": low_int}]
    unprivileged = [{"race": high_int}]

    feat_cols = [f"f{i}" for i in range(X_tr_mat.shape[1])]
    df_tr = pd.DataFrame(X_tr_mat, columns=feat_cols)
    df_tr["race"] = a_tr_int
    df_tr["y"] = y_tr
    df_te = pd.DataFrame(X_te_mat, columns=feat_cols)
    df_te["race"] = a_te_int
    df_te["y"] = y_te

    bld_tr = BinaryLabelDataset(
        df=df_tr, label_names=["y"], protected_attribute_names=["race"],
        favorable_label=1, unfavorable_label=0,
    )
    bld_te = BinaryLabelDataset(
        df=df_te, label_names=["y"], protected_attribute_names=["race"],
        favorable_label=1, unfavorable_label=0,
    )

    # Reweighing
    try:
        rw = Reweighing(unprivileged_groups=unprivileged,
                        privileged_groups=privileged)
        bld_tr_rw = rw.fit_transform(bld_tr)
        clf = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr_mat, y_tr, sample_weight=bld_tr_rw.instance_weights)
        yhat = clf.predict(X_te_mat).astype(int)
        rows.append(metrics_row("Reweighing+LR", "aif360",
                                yhat, y_te, a_te, high_g))
    except Exception as e:
        print(f"  [aif360 Reweighing] failed: {e}")

    # EqOddsPostprocessing (needs scores on validation; we use the
    # in-sample LR predictions on train for fitting, val for evaluation.)
    try:
        clf = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr_mat, y_tr)
        scores_tr = clf.predict_proba(X_tr_mat)[:, 1]
        scores_te = clf.predict_proba(X_te_mat)[:, 1]
        bld_tr_scored = bld_tr.copy(deepcopy=True)
        bld_tr_scored.scores = scores_tr.reshape(-1, 1)
        bld_te_scored = bld_te.copy(deepcopy=True)
        bld_te_scored.scores = scores_te.reshape(-1, 1)
        eq = EqOddsPostprocessing(
            unprivileged_groups=unprivileged,
            privileged_groups=privileged,
            seed=42,
        )
        eq.fit(bld_tr, bld_tr_scored)
        bld_te_pp = eq.predict(bld_te_scored)
        yhat = bld_te_pp.labels.flatten().astype(int)
        rows.append(metrics_row("EqOddsPostproc", "aif360",
                                yhat, y_te, a_te, high_g))
    except Exception as e:
        print(f"  [aif360 EqOddsPostproc] failed: {e}")
    return rows


def evaluate_aequitas(
    method_to_yhats: dict[str, np.ndarray],
    y_te: np.ndarray, a_te: np.ndarray, high_g: str,
) -> list[dict]:
    """Aequitas audit on the same predictions our other libraries produce.
    Aequitas is auditing-only, so we feed it each method's y_hat and
    record the disparity metrics it reports."""
    rows: list[dict] = []
    if not HAVE_AEQUITAS:
        return rows
    for method, yhat in method_to_yhats.items():
        try:
            df = pd.DataFrame({
                "score": yhat.astype(int),
                "label_value": y_te.astype(int),
                "race": a_te,
            })
            g = Group()
            xtab, _ = g.get_crosstabs(df)
            bias = Bias()
            bias_ref = bias.get_disparity_predefined_groups(
                xtab, original_df=df,
                ref_groups_dict={"race": high_g},
                alpha=0.05,
            )
            # Headline disparity: PPR (predicted positive rate) disparity
            # across groups
            pprs = xtab[xtab["attribute_name"] == "race"][
                ["attribute_value", "ppr"]
            ]
            ppr_max = float(pprs["ppr"].max())
            ppr_min = float(pprs["ppr"].min())
            row = {
                "library": "aequitas",
                "method": method,
                "accuracy": float(np.mean(yhat == y_te)),
                "DPD": ppr_max - ppr_min,
                "EOD": float("nan"),  # aequitas doesn't report this directly
                "OCF_violation": float("nan"),
                f"TPR_{high_g}": float(
                    xtab[(xtab["attribute_value"] == high_g)]["tpr"].iloc[0]
                ),
            }
            rows.append(row)
        except Exception as e:
            print(f"  [aequitas {method}] failed: {e}")
    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./data/brfss_2024_clean.parquet")
    ap.add_argument("--model", default="XGBoost",
                    choices=["LogisticRegression", "RandomForest", "MLP", "XGBoost"])
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.data}")
    df = pd.read_parquet(args.data).dropna(subset=[TARGET])
    print(f"  shape: {df.shape}")

    num_used = [c for c in NUMERIC if c in df.columns]
    cat_used = [c for c in CATEGORICAL if c in df.columns]
    X = df[num_used + cat_used].copy()
    y = df[TARGET].astype(int).to_numpy()
    a = df["race_eth"].astype(str).to_numpy()

    groups = sorted(list(np.unique(a)))
    pi_emp = {g: float(np.mean(y[a == g])) for g in groups}
    high_g = max(pi_emp, key=pi_emp.get)
    print(f"  high-burden group: {high_g}")

    # Train/test split
    X_tr, X_te, y_tr, y_te, a_tr, a_te = train_test_split(
        X, y, a, test_size=args.test_frac, stratify=y, random_state=args.seed
    )
    print(f"  train: {len(X_tr)}  test: {len(X_te)}")

    # Train base model on train; produce scores on train (for fitting
    # post-processors) and test (for evaluation).
    print(f"[train] {args.model}")
    pipe, scores_te = train_and_score(
        X_tr, y_tr, X_te, num_used, cat_used, args.model, seed=args.seed
    )
    scores_tr = pipe.predict_proba(X_tr)[:, 1]
    auroc_te = roc_auc_score(y_te, scores_te)
    print(f"  AUROC (test): {auroc_te:.4f}")

    all_rows: list[dict] = []

    # 1. Our methods (OCF, DP, EO, unmitigated)
    print("\n[evaluate] ocf (ours)")
    our_rows = evaluate_ocf_ours(scores_tr, y_tr, a_tr,
                                  scores_te, y_te, a_te, high_g)
    all_rows.extend(our_rows)
    for r in our_rows:
        print(f"  {r['method']:<30} acc={r['accuracy']:.4f} "
              f"DPD={r['DPD']:.4f} OCFviol={r['OCF_violation']:.4f}")

    # 2. Fairlearn
    print("\n[evaluate] fairlearn")
    fl_rows = evaluate_fairlearn(
        scores_tr, y_tr, a_tr, scores_te, y_te, a_te,
        X_tr, X_te, pipe, high_g,
    )
    all_rows.extend(fl_rows)
    for r in fl_rows:
        print(f"  {r['method']:<30} acc={r['accuracy']:.4f} "
              f"DPD={r['DPD']:.4f} OCFviol={r['OCF_violation']:.4f}")

    # 3. AIF360
    print("\n[evaluate] aif360")
    aif_rows = evaluate_aif360(X_tr, y_tr, a_tr, X_te, y_te, a_te,
                                num_used, cat_used, high_g)
    all_rows.extend(aif_rows)
    for r in aif_rows:
        print(f"  {r['method']:<30} acc={r['accuracy']:.4f} "
              f"DPD={r['DPD']:.4f} OCFviol={r['OCF_violation']:.4f}")

    # 4. Aequitas — audit our methods' predictions
    method_yhats = {
        r["method"]: None for r in our_rows
    }
    if HAVE_AEQUITAS:
        method_yhats = {}
        # Re-derive y_hats for OCF/DP/Unmit
        method_yhats["OCF"] = ocf.predict_ocf(
            scores_te, a_te,
            ocf.fit_ocf(scores_tr, a_tr, y_tr),
        )
        method_yhats["DP"] = ocf.predict_dp(
            scores_te, a_te,
            ocf.fit_dp(scores_tr, a_tr, y=y_tr),
        )
        method_yhats["Unmitigated"] = ocf.predict_unmitigated(
            scores_te,
            ocf.fit_unmitigated(scores_tr, y_tr),
        )
        print("\n[evaluate] aequitas (auditing our predictions)")
        aeq_rows = evaluate_aequitas(method_yhats, y_te, a_te, high_g)
        all_rows.extend(aeq_rows)
        for r in aeq_rows:
            print(f"  {r['method']:<30} acc={r['accuracy']:.4f} "
                  f"DPD={r['DPD']:.4f}")

    # Save and summary
    out_df = pd.DataFrame(all_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{args.model}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  saved: {out_path}")

    print("\n" + "=" * 90)
    print(f"BENCHMARK SUMMARY  ({args.model},  high-burden: {high_g})")
    print("=" * 90)
    print(f"{'Library':<14} {'Method':<28} {'Acc':>7} {'DPD':>7} "
          f"{'EOD':>7} {'OCFv':>7} {'TPR_h':>7}")
    for _, r in out_df.iterrows():
        print(f"{r['library']:<14} {r['method']:<28} "
              f"{r['accuracy']:>7.4f} {r['DPD']:>7.4f} {r['EOD']:>7.4f} "
              f"{r.get('OCF_violation', np.nan):>7.4f} "
              f"{r.get(f'TPR_{high_g}', np.nan):>7.4f}")

    # Cross-check: ocf.fit_dp ≡ fairlearn ThresholdOptimizer(DP)
    print("\n[sanity] ocf.fit_dp should give same DPD as fairlearn ThresholdOpt(DP)")
    ocf_dp = out_df[(out_df["library"] == "ocf (ours)") &
                    (out_df["method"] == "DP")]
    fl_dp = out_df[(out_df["library"] == "fairlearn") &
                   (out_df["method"] == "DP (ThresholdOpt)")]
    if len(ocf_dp) and len(fl_dp):
        print(f"  ocf DP DPD       = {float(ocf_dp.iloc[0]['DPD']):.6f}")
        print(f"  fairlearn DP DPD = {float(fl_dp.iloc[0]['DPD']):.6f}")
        gap = abs(float(ocf_dp.iloc[0]['DPD']) - float(fl_dp.iloc[0]['DPD']))
        ok = "✓" if gap < 1e-2 else "⚠"
        print(f"  agreement: {ok}  (gap={gap:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
