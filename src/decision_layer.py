"""
decision_layer.py
=================
The decision-intelligence layer that distinguishes this study from prior
TAT-prediction work. Two components:

1. EXPLAINABILITY (SHAP)
   Feature attributions for the best TAT-breach model, so a laboratory manager
   sees *why* a sample is flagged high-risk and can act on the driver
   (e.g., reallocate staff when workload-per-staff is the dominant contributor).

2. COST-OF-QUALITY DECISION OPTIMIZATION
   Predictions are converted into an operational decision -- whether to expedite
   (intervene on) a sample -- under an explicit cost model. We sweep the decision
   threshold, compute expected total cost, and compare the model-guided policy to
   managerial baselines (treat-none, treat-all, priority-only rule). This yields
   the economic payoff a business audience cares about, plus a sensitivity
   analysis over the cost ratio.

Cost model (illustrative unit costs; varied in sensitivity analysis):
   C_intervene -- marginal cost of flagging + expediting one sample (staff time).
   C_breach    -- expected downstream cost of one unmitigated TAT breach
                  (delayed result: extended length of stay, repeat orders,
                  clinician follow-up). For time-critical donor screening the
                  effective C_breach is far higher, which the sensitivity curve
                  makes explicit.
Intervening on a predicted-positive sample is assumed to mitigate a true breach.

Outputs (results/metrics/):
   shap_importance_tat_breach.csv   -- mean|SHAP| per feature
   shap_values_tat_breach.npz       -- SHAP values + data for the beeswarm figure
   cost_threshold_sweep.csv         -- expected cost across thresholds
   cost_strategy_comparison.csv     -- policy comparison table (Table 2)
   cost_sensitivity.csv             -- optimal cost across C_breach/C_intervene ratios

Author: Jaisritha Damavarapu
License: MIT
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
METRICS_DIR = os.path.join(ROOT, "results", "metrics")

# Illustrative unit costs (USD). Varied in the sensitivity analysis.
C_INTERVENE = 15.0
C_BREACH = 120.0


def load_bundle(task: str):
    with open(os.path.join(METRICS_DIR, f"model_bundle_{task}.pkl"), "rb") as fh:
        return pickle.load(fh)


def densify(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def compute_shap(bundle):
    """Compute SHAP values on the held-out test set for the best model."""
    pipe = bundle["pipe"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    name = bundle["best_name"]

    X_tr_t = densify(pre.transform(bundle["X_tr"]))
    X_te_t = densify(pre.transform(bundle["X_te"]))
    feat = list(pre.get_feature_names_out())
    # Tidy feature names for presentation
    feat = [f.replace("cat__", "").replace("num__", "") for f in feat]

    if name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_te_t)
        if isinstance(sv, list):  # some versions return per-class list
            sv = sv[1]
    else:  # LogisticRegression / linear
        explainer = shap.LinearExplainer(clf, X_tr_t)
        sv = explainer.shap_values(X_te_t)

    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, f, classes)
        sv = sv[:, :, -1]

    mean_abs = np.abs(sv).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(os.path.join(METRICS_DIR, f"shap_importance_{bundle['task']}.csv"), index=False)
    np.savez_compressed(
        os.path.join(METRICS_DIR, f"shap_values_{bundle['task']}.npz"),
        shap_values=sv, data=X_te_t, feature_names=np.array(feat, dtype=object),
    )
    print(f"SHAP ({name}) top drivers for {bundle['task']}:")
    print(imp.head(8).to_string(index=False))
    return imp


def confusion_at(y_true, y_prob, t):
    pred = (y_prob >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def policy_cost(tp, fp, fn, c_int=C_INTERVENE, c_breach=C_BREACH):
    """Intervene on predicted positives (TP+FP); miss breaches among FN."""
    return c_int * (tp + fp) + c_breach * fn


def threshold_sweep(y_true, y_prob):
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        tp, fp, fn, tn = confusion_at(y_true, y_prob, t)
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_cost": policy_cost(tp, fp, fn),
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(METRICS_DIR, "cost_threshold_sweep.csv"), index=False)
    return sweep


def strategy_comparison(df_pred, sweep):
    """Compare managerial policies on the held-out test set."""
    y = df_pred["y_true"].values
    p = df_pred["y_prob"].values
    n = len(y)
    pos = int(y.sum())

    # Baseline 1: treat none -> every actual breach is unmitigated
    cost_none = policy_cost(tp=0, fp=0, fn=pos)
    # Baseline 2: treat all -> intervene on everyone, no missed breaches
    cost_all = policy_cost(tp=pos, fp=n - pos, fn=0)
    # Baseline 3: priority-only rule -> intervene on STAT samples only
    is_stat = (df_pred["priority"] == "STAT").values
    tp = int(((is_stat) & (y == 1)).sum())
    fp = int(((is_stat) & (y == 0)).sum())
    fn = int(((~is_stat) & (y == 1)).sum())
    cost_rule = policy_cost(tp, fp, fn)
    # Model at default 0.5 and at cost-optimal threshold
    tp5, fp5, fn5, _ = confusion_at(y, p, 0.5)
    cost_05 = policy_cost(tp5, fp5, fn5)
    opt = sweep.loc[sweep["total_cost"].idxmin()]
    cost_opt = opt["total_cost"]
    t_opt = opt["threshold"]

    def per_1k(c):
        return c / n * 1000.0

    rows = [
        ("Treat none (no intervention)", cost_none, "-"),
        ("Treat all (expedite every sample)", cost_all, "-"),
        ("Priority rule (expedite STAT only)", cost_rule, "-"),
        ("Model @ threshold 0.50", cost_05, "0.50"),
        ("Model @ cost-optimal threshold", cost_opt, f"{t_opt:.2f}"),
    ]
    best_baseline = min(cost_none, cost_all, cost_rule)
    out = []
    for label, cost, thr in rows:
        out.append({
            "strategy": label,
            "threshold": thr,
            "total_cost_test": round(cost, 2),
            "cost_per_1000_samples": round(per_1k(cost), 2),
            "savings_vs_treat_none_pct": round((cost_none - cost) / cost_none * 100, 1),
            "savings_vs_best_baseline_pct": round((best_baseline - cost) / best_baseline * 100, 1),
        })
    comp = pd.DataFrame(out)
    comp.to_csv(os.path.join(METRICS_DIR, "cost_strategy_comparison.csv"), index=False)
    print("\nCost-of-quality policy comparison (per held-out test set):")
    print(comp.to_string(index=False))
    return comp, t_opt


def sensitivity(df_pred):
    """Optimal expected cost as a function of the C_breach/C_intervene ratio."""
    y = df_pred["y_true"].values
    p = df_pred["y_prob"].values
    n = len(y)
    pos = int(y.sum())
    rows = []
    for ratio in [2, 4, 6, 8, 10, 15, 20, 30, 50]:
        c_int, c_breach = 15.0, 15.0 * ratio
        best_cost, best_t = np.inf, None
        for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
            tp, fp, fn, _ = confusion_at(y, p, t)
            c = c_int * (tp + fp) + c_breach * fn
            if c < best_cost:
                best_cost, best_t = c, t
        cost_none = c_breach * pos
        rows.append({
            "cost_ratio_breach_to_intervene": ratio,
            "optimal_threshold": best_t,
            "optimal_cost": round(best_cost, 2),
            "treat_none_cost": round(cost_none, 2),
            "savings_pct": round((cost_none - best_cost) / cost_none * 100, 1),
        })
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(METRICS_DIR, "cost_sensitivity.csv"), index=False)
    print("\nCost-ratio sensitivity (optimal threshold shifts with stakes):")
    print(sens.to_string(index=False))
    return sens


def main():
    # --- Explainability on the TAT-breach model (primary task) ---
    bundle = load_bundle("tat_breach")
    compute_shap(bundle)
    # Also attributions for rejection (secondary) for completeness
    compute_shap(load_bundle("rejection"))

    # --- Cost-of-quality decision optimization on TAT-breach ---
    df_pred = pd.read_csv(os.path.join(METRICS_DIR, "best_model_predictions_tat_breach.csv"))
    sweep = threshold_sweep(df_pred["y_true"].values, df_pred["y_prob"].values)
    strategy_comparison(df_pred, sweep)
    sensitivity(df_pred)
    print(f"\nDecision-layer outputs written to {METRICS_DIR}")


if __name__ == "__main__":
    main()
