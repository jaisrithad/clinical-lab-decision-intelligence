"""
make_tables.py
=============
Build the two publication tables from actual experimental outputs, emitting
both LaTeX (for the IEEE manuscript) and CSV (for the repository).

Table 1 -- Model comparison across both tasks: 5-fold CV ROC-AUC, held-out
           test ROC-AUC, PR-AUC, Brier score, and F1 at the 0.5 threshold.
Table 2 -- Cost-of-quality policy comparison: operational cost and savings of
           each decision policy on the held-out test set.

Author: Jaisritha Damavarapu
License: MIT
"""

from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
METRICS_DIR = os.path.join(ROOT, "results", "metrics")
TABLE_DIR = os.path.join(ROOT, "results", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

TASK_LABEL = {"rejection": "Specimen rejection", "tat_breach": "TAT breach"}


def table1():
    cv = pd.read_csv(os.path.join(METRICS_DIR, "cv_metrics.csv"))
    te = pd.read_csv(os.path.join(METRICS_DIR, "test_metrics.csv"))
    m = cv.merge(te, on=["task", "model"], suffixes=("_cv", ""))
    m["Task"] = m["task"].map(TASK_LABEL)
    m["CV ROC-AUC"] = m.apply(lambda r: f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}", axis=1)
    m["Test ROC-AUC"] = m["roc_auc"].map(lambda v: f"{v:.3f}")
    m["PR-AUC"] = m["pr_auc"].map(lambda v: f"{v:.3f}")
    m["Brier"] = m["brier"].map(lambda v: f"{v:.3f}")
    m["F1"] = m["f1"].map(lambda v: f"{v:.3f}")
    out = m[["Task", "model", "CV ROC-AUC", "Test ROC-AUC", "PR-AUC", "Brier", "F1"]]
    out = out.rename(columns={"model": "Model"})
    out = out.sort_values(["Task", "Model"]).reset_index(drop=True)

    out.to_csv(os.path.join(TABLE_DIR, "table1_model_comparison.csv"), index=False)
    with open(os.path.join(TABLE_DIR, "table1_model_comparison.tex"), "w") as fh:
        fh.write(_latex(out, "tab:models",
                        "Predictive performance across models and tasks "
                        "(mean $\\pm$ s.d. over 5-fold cross-validation; test "
                        "metrics on a held-out 25\\% split).",
                        col_fmt="llccccc"))
    print("Table 1 (model comparison):")
    print(out.to_string(index=False))
    return out


def table2():
    comp = pd.read_csv(os.path.join(METRICS_DIR, "cost_strategy_comparison.csv"))
    comp = comp.rename(columns={
        "strategy": "Decision policy",
        "threshold": "Threshold",
        "total_cost_test": "Total cost (USD)",
        "cost_per_1000_samples": "Cost / 1000 samples",
        "savings_vs_treat_none_pct": "Savings vs. treat-none (\\%)",
        "savings_vs_best_baseline_pct": "Savings vs. best baseline (\\%)",
    })
    comp.to_csv(os.path.join(TABLE_DIR, "table2_cost_comparison.csv"), index=False)
    with open(os.path.join(TABLE_DIR, "table2_cost_comparison.tex"), "w") as fh:
        fh.write(_latex(comp, "tab:cost",
                        "Cost-of-quality comparison of decision policies on the "
                        "held-out test set (unit costs: \\$15 intervene, \\$120 "
                        "per unmitigated breach).", col_fmt="lccccc"))
    print("\nTable 2 (cost-of-quality comparison):")
    print(comp.to_string(index=False))
    return comp


def _latex(df, label, caption, col_fmt):
    header = " & ".join(df.columns) + " \\\\"
    body = " \\\\\n".join(" & ".join(str(v) for v in row) for row in df.values) + " \\\\"
    return (
        "\\begin{table*}[!t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n\\hline\n"
        f"{header}\n\\hline\n{body}\n\\hline\n"
        "\\end{tabular}\n\\end{table*}\n"
    )


if __name__ == "__main__":
    table1()
    table2()
    print(f"\nTables written to {TABLE_DIR}")
