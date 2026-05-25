#!/usr/bin/env python3
"""
Main experiment: train 4 ML baselines on BRFSS 2024, apply 5 fairness
conditions (Unmitigated, DP, EO, Calibration, OCF), report fairness
and accuracy metrics across race/ethnicity subgroups.

This is the cross-validation pipeline producing the headline
``all_metrics.csv``, ``summary.csv``, and ``headline_table.csv`` files
reported in Section 5 of the paper. For bootstrap CIs, use
``04_bootstrap_audit.py``.

ML baselines
------------
1. Logistic Regression
2. Random Forest
3. Multi-Layer Perceptron
4. XGBoost (if installed)

Fairness conditions (post-processing on each baseline's scores)
---------------------------------------------------------------
- Unmitigated (single global threshold)
- DP (group thresholds matching mean(y))
- EO (HPS-LP-based, randomized; ``ocf.eo_hardt``)
- Calibration (per-group isotonic + global threshold)
- OCF (Algorithm 1; rho_a = c * pi_a)

Usage
-----
    python 03_train_and_audit.py \\
        --data data/brfss_2024_clean.parquet \\
        --n-folds 5
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
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, HERE.parent.parent, Path.cwd()):
    if (cand / "ocf" / "__init__.py").exists():
        sys.path.insert(0, str(cand))
        break

import ocf  # noqa: E402

NUMERIC_FEATURES = [
    "male", "has_insurance", "regular_provider", "cost_barrier",
    "binge_drink", "current_smoker", "heavy_drink", "meets_aerobic",
    "gen_health", "diabetes", "metro",
]
CATEGORICAL_FEATURES = [
    "age_group", "education", "income", "marital", "region",
]
TARGET = "hiv_tested_ever"


def build_design_matrix(df: pd.DataFrame) -> tuple:
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    num_used = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_used = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[num_used + cat_used].copy()
    y = df[TARGET].astype(int).values
    a_dict: dict[str, np.ndarray] = {}
    if "race_eth" in df.columns:
        a_dict["race_eth"] = df["race_eth"].astype(str).values
    if "msm" in df.columns:
        a_dict["msm"] = df["msm"].values
    return X, y, a_dict, num_used, cat_used


def make_preprocessor(num_used, cat_used):
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


def get_classifiers(random_state: int = 42) -> dict:
    out = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, C=1.0, random_state=random_state, solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200,
            early_stopping=True, random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
    except ImportError:
        print("  [info] xgboost not installed — skipping XGBoost.")
    return out


def audit_one_model(name, model, X, y, a, num_used, cat_used,
                    n_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    print(f"\n--- Auditing {name} ---")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_rows: list[dict] = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        ate = a[te]
        pre = make_preprocessor(num_used, cat_used)
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        scores = pipe.predict_proba(Xte)[:, 1]
        auroc = float(roc_auc_score(yte, scores))
        brier = float(brier_score_loss(yte, scores))
        result = ocf.evaluate_all(scores, yte, ate, seed=random_state + fold)
        for row in result["rows"]:
            row["fold"] = fold
            row["model"] = name
            row["auroc"] = auroc
            row["brier"] = brier
            all_rows.append(row)
        print(f"  fold {fold}: AUROC={auroc:.4f}  Brier={brier:.4f}")
    return pd.DataFrame(all_rows)


def summarize_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in audit_df.columns
                   if c not in ("fold", "model", "method")]
    rows = []
    for (model, method), grp in audit_df.groupby(["model", "method"]):
        row: dict = {"model": model, "method": method}
        for col in metric_cols:
            vals = grp[col].dropna().to_numpy()
            if len(vals):
                try:
                    row[col + "_mean"] = float(np.mean(vals))
                    row[col + "_std"] = float(np.std(vals))
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    return pd.DataFrame(rows)


def make_headline_table(audit_df: pd.DataFrame, high_g: str) -> pd.DataFrame:
    rows = []
    for (model, method), grp in audit_df.groupby(["model", "method"]):
        tpr_col = f"TPR_{high_g}"
        if tpr_col not in grp.columns:
            continue
        rows.append({
            "model": model,
            "method": method,
            "auroc_mean": grp["auroc"].mean(),
            "accuracy_mean": grp["accuracy"].mean(),
            "DPD_mean": grp["DPD"].mean(),
            "EOD_mean": grp["EOD"].mean(),
            "OCF_violation_mean": grp["OCF_violation"].mean(),
            f"TPR_{high_g}_mean": grp[tpr_col].mean(),
            f"rho_{high_g}_mean": grp[f"rho_{high_g}"].mean(),
        })
    df = pd.DataFrame(rows)
    out_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        if "Unmitigated" not in sub["method"].values:
            continue
        baseline = float(sub.loc[sub["method"] == "Unmitigated",
                                f"TPR_{high_g}_mean"].iloc[0])
        sub[f"d_TPR_{high_g}_vs_unmit"] = sub[f"TPR_{high_g}_mean"] - baseline
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./data/brfss_2024_clean.parquet")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--protected", default="race_eth", choices=["race_eth", "msm"])
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--high-burden", default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  shape: {df.shape}")
    if args.sample_frac < 1.0:
        n_orig = len(df)
        df = df.sample(frac=args.sample_frac, random_state=42)
        print(f"  subsampled: {len(df):,} from {n_orig:,}")
    if args.protected == "msm" and "msm" not in df.columns:
        print("ERROR: MSM column not present.")
        return 1
    if args.protected == "msm":
        df = df[df["msm"].notna()].copy()

    X, y, a_dict, num_used, cat_used = build_design_matrix(df)
    a_arr = a_dict[args.protected]
    groups = list(np.unique(a_arr))
    if args.high_burden is None:
        base_rates = {g: float(np.mean(y[a_arr == g])) for g in groups}
        high_burden = max(base_rates, key=base_rates.get)
        print(f"  auto-detected high-burden group: '{high_burden}' "
              f"(pi={base_rates[high_burden]:.4f})")
    else:
        high_burden = args.high_burden

    print(f"\n[setup]  protected={args.protected}, groups={groups}")
    for g in groups:
        print(f"    {g:15s}: n={int((a_arr == g).sum()):7,d}  "
              f"pi={float(np.mean(y[a_arr == g])):.4f}")

    classifiers = get_classifiers()
    audits = [
        audit_one_model(name, model, X, y, a_arr, num_used, cat_used,
                       n_folds=args.n_folds)
        for name, model in classifiers.items()
    ]
    full = pd.concat(audits, ignore_index=True)
    full.to_csv(out_dir / "all_metrics.csv", index=False)
    print(f"\n  saved: {out_dir / 'all_metrics.csv'}")

    summary = summarize_audit(full)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"  saved: {out_dir / 'summary.csv'}")

    headline = make_headline_table(full, high_burden)
    headline.to_csv(out_dir / "headline_table.csv", index=False)
    print(f"  saved: {out_dir / 'headline_table.csv'}")

    print("\n" + "=" * 78)
    print(f"HEADLINE: TPR change for '{high_burden}' (highest-burden group)")
    print("=" * 78)
    display_cols = ["model", "method",
                   f"TPR_{high_burden}_mean", f"d_TPR_{high_burden}_vs_unmit",
                   "DPD_mean", "OCF_violation_mean", "accuracy_mean"]
    available = [c for c in display_cols if c in headline.columns]
    print(headline[available].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("THEOREM VERIFICATION ON BRFSS 2024 (cross-validation)")
    print("=" * 78)
    for model in headline["model"].unique():
        sub = headline[headline["model"] == model]
        col = f"d_TPR_{high_burden}_vs_unmit"
        if col not in sub.columns:
            continue
        dp_change = float(sub.loc[sub["method"] == "DP", col].iloc[0])
        ocf_change = float(sub.loc[sub["method"] == "OCF", col].iloc[0])
        thm1 = "✓" if dp_change < 0 else "✗"
        thm4 = "✓" if ocf_change > dp_change else "✗"
        print(f"  {model:20s}:  Thm1 DP-Harm: {thm1} "
              f"(ΔTPR={dp_change:+.4f})    "
              f"Thm4 OCF>DP: {thm4} "
              f"(ΔTPR={ocf_change:+.4f} > {dp_change:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
