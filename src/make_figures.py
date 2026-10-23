"""
make_figures.py
==============
Generate the two main publication figures and a programmatic graphical abstract
from actual experimental outputs. No AI image generation is used: every element
is drawn by matplotlib from the saved metrics.

Figure 1 -- Predictive performance: ROC and Precision-Recall curves for the
            three models on the TAT-breach task (held-out test set).
Figure 2 -- Decision intelligence: (a) SHAP driver importance for the TAT-breach
            model; (b) expected cost vs decision threshold with the cost-optimal
            point and managerial baselines.
Graphical abstract -- single-panel visual summary of the framework + headline
            cost saving.

All figures saved to results/figures/ as 300-dpi PNG and vector PDF.

Author: Jaisritha Damavarapu
License: MIT
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

import pickle
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from experiments import build_preprocessor, make_models, scale_pos_weight  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
METRICS_DIR = os.path.join(ROOT, "results", "metrics")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})
COLORS = {"LogisticRegression": "#1b4965", "RandomForest": "#5fa8d3", "XGBoost": "#cae9ff"}
ACCENT = "#bc4749"
OPT = "#2a9d8f"


def _bundle(task):
    with open(os.path.join(METRICS_DIR, f"model_bundle_{task}.pkl"), "rb") as fh:
        return pickle.load(fh)


def refit_all_models(task):
    """Refit the three models on the saved split; return probs on the test set."""
    b = _bundle(task)
    X_tr, X_te, y_tr, y_te = b["X_tr"], b["X_te"], b["y_tr"], b["y_te"]
    probs = {}
    for name, (est, needs_scaling) in make_models().items():
        e = est
        if name == "XGBoost":
            e = XGBClassifier(**{**est.get_params(), "scale_pos_weight": scale_pos_weight(y_tr)})
        pipe = Pipeline([("pre", build_preprocessor(needs_scaling)), ("clf", e)])
        pipe.fit(X_tr, y_tr)
        probs[name] = pipe.predict_proba(X_te)[:, 1]
    return y_te.values, probs


def figure1():
    y, probs = refit_all_models("tat_breach")
    test = pd.read_csv(os.path.join(METRICS_DIR, "test_metrics.csv"))
    test = test[test["task"] == "tat_breach"].set_index("model")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=COLORS[name], lw=2,
                 label=f"{name} (AUC={test.loc[name,'roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("(a) ROC curves — TAT-breach")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)

    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=COLORS[name], lw=2,
                 label=f"{name} (AP={test.loc[name,'pr_auc']:.3f})")
    ax2.axhline(y.mean(), color="k", ls="--", lw=0.8, alpha=0.6,
                label=f"Prevalence={y.mean():.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision–Recall curves — TAT-breach")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"figure1_performance.{ext}"))
    plt.close(fig)
    print("Figure 1 saved (ROC + PR).")


def figure2():
    imp = pd.read_csv(os.path.join(METRICS_DIR, "shap_importance_tat_breach.csv")).head(10)
    sweep = pd.read_csv(os.path.join(METRICS_DIR, "cost_threshold_sweep.csv"))
    comp = pd.read_csv(os.path.join(METRICS_DIR, "cost_strategy_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # (a) SHAP importance
    order = imp.iloc[::-1]
    ax1.barh(order["feature"], order["mean_abs_shap"], color="#1b4965")
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.set_title("(a) TAT-breach risk drivers (SHAP)")

    # (b) Cost vs threshold
    ax2.plot(sweep["threshold"], sweep["total_cost"], color="#1b4965", lw=2, label="Model policy")
    opt = sweep.loc[sweep["total_cost"].idxmin()]
    ax2.scatter([opt["threshold"]], [opt["total_cost"]], color=OPT, zorder=5, s=45,
                label=f"Optimal t={opt['threshold']:.2f}")
    # Baselines as horizontal reference lines
    def cost_of(label):
        return comp.loc[comp["strategy"].str.contains(label), "total_cost_test"].values[0]
    ax2.axhline(cost_of("Treat none"), color=ACCENT, ls="--", lw=1, label="Treat none")
    ax2.axhline(cost_of("Priority rule"), color="#e9c46a", ls="--", lw=1, label="Priority-only rule")
    ax2.set_xlabel("Decision threshold"); ax2.set_ylabel("Expected total cost (USD, test set)")
    ax2.set_title("(b) Cost-of-quality optimization")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"figure2_decision_intelligence.{ext}"))
    plt.close(fig)
    print("Figure 2 saved (SHAP + cost).")


def graphical_abstract():
    comp = pd.read_csv(os.path.join(METRICS_DIR, "cost_strategy_comparison.csv"))
    opt_row = comp.loc[comp["strategy"].str.contains("cost-optimal")].iloc[0]
    saving = opt_row["savings_vs_best_baseline_pct"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    stages = [
        ("Lab\noperational\ndata", "#cae9ff"),
        ("Explainable\nML models\n(LR/RF/XGB)", "#5fa8d3"),
        ("SHAP\nrisk drivers", "#1b4965"),
        ("Cost-of-quality\ndecision\noptimization", "#2a9d8f"),
    ]
    x = 0.4
    for i, (label, color) in enumerate(stages):
        box = FancyBboxPatch((x, 3.4), 1.9, 1.6, boxstyle="round,pad=0.08",
                             fc=color, ec="none")
        ax.add_patch(box)
        tc = "white" if color in ("#1b4965", "#2a9d8f") else "#0b2545"
        ax.text(x + 0.95, 4.2, label, ha="center", va="center", fontsize=9,
                color=tc, weight="bold")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.95, 4.2), (x + 2.4, 4.2),
                         arrowstyle="-|>", mutation_scale=16, color="#5a6472", lw=1.5))
        x += 2.35

    ax.text(5.0, 2.5,
            "Explainable Decision Intelligence for Clinical Laboratory Operations",
            ha="center", va="center", fontsize=12, weight="bold", color="#0b2545")
    ax.text(5.0, 1.75,
            "Predicting turnaround-time breaches & specimen rejection for cost-aware quality management",
            ha="center", va="center", fontsize=9.5, color="#333333", style="italic")

    # Headline result badge
    badge = FancyBboxPatch((3.1, 0.25), 3.8, 1.05, boxstyle="round,pad=0.08",
                           fc="#f2e8cf", ec="#2a9d8f", lw=1.5)
    ax.add_patch(badge)
    ax.text(5.0, 0.78, f"Up to {saving:.0f}% lower operational cost\nvs. best managerial heuristic",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#2a6f5f")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"graphical_abstract.{ext}"))
    plt.close(fig)
    print("Graphical abstract saved.")


if __name__ == "__main__":
    figure1()
    figure2()
    graphical_abstract()
    print(f"\nAll figures written to {FIG_DIR}")
