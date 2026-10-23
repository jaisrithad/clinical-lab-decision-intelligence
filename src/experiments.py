"""
experiments.py
==============
Train and evaluate baseline models for two clinical-laboratory operational
decision tasks, with leakage-safe features available at accessioning time:

    Task A -- specimen rejection (pre-analytical quality failure)
    Task B -- turnaround-time (TAT) breach on processed samples

Models: Logistic Regression, Random Forest, XGBoost.
Validation: stratified 5-fold cross-validation for ROC-AUC / PR-AUC / Brier,
plus a held-out stratified test set for final reported metrics.

Outputs (written to results/):
    metrics/cv_metrics.csv          -- per-model, per-task CV mean +/- std
    metrics/test_metrics.csv        -- per-model, per-task held-out test metrics
    metrics/best_model_predictions_*.csv -- test-set predictions for best model
    metrics/feature_matrix_*.pkl    -- processed train/test matrices for SHAP

Author: Jaisritha Damavarapu
License: MIT
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "lab_operations.csv")
METRICS_DIR = os.path.join(ROOT, "results", "metrics")

CATEGORICAL = ["priority", "section", "specimen_type", "collection_source"]
NUMERIC = [
    "hour_of_day", "pending_samples", "staff_on_duty", "workload_per_staff",
    "analyzer_utilization", "hemolysis_index", "transport_time_min",
]
BINARY = [
    "is_night_shift", "is_weekend", "is_addon", "is_repeat",
    "label_complete", "volume_adequate",
]
# Columns that leak the label or are posterior to the decision point
LEAKY = ["turnaround_time_min", "tat_threshold_min", "tat_breach", "specimen_rejected"]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot categoricals; optionally standardize numerics (for LR)."""
    num_tf = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
            ("num", num_tf, NUMERIC + BINARY),
        ]
    )


def make_models():
    """Return dict of model_name -> (estimator, needs_scaling)."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            True,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=SEED,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                n_jobs=-1, random_state=SEED,
            ),
            False,
        ),
    }


def get_task_frame(df: pd.DataFrame, task: str):
    """Return (X, y) for the requested task with leakage-safe features."""
    features = CATEGORICAL + NUMERIC + BINARY
    if task == "rejection":
        y = df["specimen_rejected"].astype(int)
        X = df[features].copy()
    elif task == "tat_breach":
        proc = df[df["specimen_rejected"] == 0].dropna(subset=["tat_breach"])
        y = proc["tat_breach"].astype(int)
        X = proc[features].copy()
    else:
        raise ValueError(task)
    return X, y


def scale_pos_weight(y) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return neg / max(pos, 1)


def cross_validate(X, y, name, estimator, needs_scaling):
    """Stratified 5-fold CV -> mean/std of ROC-AUC, PR-AUC, Brier."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    roc, pr, brier = [], [], []
    for tr, va in skf.split(X, y):
        pre = build_preprocessor(needs_scaling)
        est = estimator
        if name == "XGBoost":
            est = XGBClassifier(**{**estimator.get_params(), "scale_pos_weight": scale_pos_weight(y.iloc[tr])})
        pipe = Pipeline([("pre", pre), ("clf", est)])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        roc.append(roc_auc_score(y.iloc[va], p))
        pr.append(average_precision_score(y.iloc[va], p))
        brier.append(brier_score_loss(y.iloc[va], p))
    return {
        "roc_auc_mean": np.mean(roc), "roc_auc_std": np.std(roc),
        "pr_auc_mean": np.mean(pr), "pr_auc_std": np.std(pr),
        "brier_mean": np.mean(brier), "brier_std": np.std(brier),
    }


def evaluate_test(X_tr, X_te, y_tr, y_te, name, estimator, needs_scaling):
    """Fit on train, evaluate on held-out test at the 0.5 threshold."""
    pre = build_preprocessor(needs_scaling)
    est = estimator
    if name == "XGBoost":
        est = XGBClassifier(**{**estimator.get_params(), "scale_pos_weight": scale_pos_weight(y_tr)})
    pipe = Pipeline([("pre", pre), ("clf", est)])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
        "brier": brier_score_loss(y_te, p),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
    }
    return metrics, pipe, p


def run_task(df, task):
    print(f"\n=== Task: {task} ===")
    X, y = get_task_frame(df, task)
    print(f"n={len(X):,}  positives={int(y.sum()):,}  prevalence={y.mean():.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    cv_rows, test_rows = [], []
    best = {"name": None, "roc": -1, "pipe": None, "p": None}
    for name, (est, needs_scaling) in make_models().items():
        cv = cross_validate(X_tr, y_tr, name, est, needs_scaling)
        cv_rows.append({"task": task, "model": name, **cv})
        tm, pipe, p = evaluate_test(X_tr, X_te, y_tr, y_te, name, est, needs_scaling)
        test_rows.append({"task": task, "model": name, **tm})
        print(f"  {name:20s} CV ROC-AUC={cv['roc_auc_mean']:.3f}+/-{cv['roc_auc_std']:.3f}"
              f"  test ROC-AUC={tm['roc_auc']:.3f}  PR-AUC={tm['pr_auc']:.3f}")
        if tm["roc_auc"] > best["roc"]:
            best = {"name": name, "roc": tm["roc_auc"], "pipe": pipe, "p": p}

    # Persist best-model test predictions and processed matrices for SHAP / cost
    os.makedirs(METRICS_DIR, exist_ok=True)
    pred_df = X_te.copy()
    pred_df["y_true"] = y_te.values
    pred_df["y_prob"] = best["p"]
    pred_df.to_csv(os.path.join(METRICS_DIR, f"best_model_predictions_{task}.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"model_bundle_{task}.pkl"), "wb") as fh:
        pickle.dump(
            {"task": task, "best_name": best["name"], "pipe": best["pipe"],
             "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te},
            fh,
        )
    print(f"  -> best model: {best['name']} (test ROC-AUC={best['roc']:.3f})")
    return cv_rows, test_rows


def main():
    df = pd.read_csv(DATA)
    all_cv, all_test = [], []
    for task in ["rejection", "tat_breach"]:
        cv_rows, test_rows = run_task(df, task)
        all_cv += cv_rows
        all_test += test_rows

    os.makedirs(METRICS_DIR, exist_ok=True)
    pd.DataFrame(all_cv).to_csv(os.path.join(METRICS_DIR, "cv_metrics.csv"), index=False)
    pd.DataFrame(all_test).to_csv(os.path.join(METRICS_DIR, "test_metrics.csv"), index=False)
    print(f"\nSaved CV and test metrics to {METRICS_DIR}")


if __name__ == "__main__":
    main()
