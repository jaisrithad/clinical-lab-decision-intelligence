# Explainable Decision Intelligence for Clinical Laboratory Operations

Predicting turnaround-time (TAT) breaches and pre-analytical specimen rejection,
and turning those predictions into **cost-rational, explainable operating
decisions** for clinical-laboratory and diagnostic-service management.

This repository accompanies the manuscript *"Explainable Decision Intelligence
for Clinical Laboratory Operations: Predicting Turnaround-Time Breaches and
Specimen Rejection for Cost-Aware Quality Management"* (prepared for IEEE Access).
Every figure and table in the paper is reproducible from the code here with a
fixed random seed.

---

## What this project does

1. **Generates** a literature-calibrated synthetic dataset of 20,000 laboratory
   test orders (real sample-level lab data is not public; the generator is
   documented and reproducible).
2. **Benchmarks** three models — Logistic Regression, Random Forest, XGBoost — on
   two leakage-safe decision tasks:
   - **Task A – Specimen rejection** (pre-analytical quality failure)
   - **Task B – TAT breach** (turnaround time exceeds the priority-specific target)
3. **Explains** the TAT-breach model with SHAP, attributing risk to
   manager-controllable drivers.
4. **Optimizes** an expedite-or-not decision under an explicit cost-of-quality
   model, and compares it to managerial baselines.

## Headline results

| Result | Value |
|---|---|
| TAT-breach prediction (best test ROC-AUC) | **0.886** |
| Specimen-rejection prediction (best test ROC-AUC) | 0.702 |
| Cost reduction vs. no intervention | **35.9%** |
| Cost reduction vs. best managerial heuristic | **23.7%** |
| Cost reduction in high-stakes regime (e.g. donor screening) | up to **77.9%** |

## Repository structure

```
clinical-lab-decision-intelligence/
├── src/
│   ├── generate_data.py      # calibrated synthetic data generator
│   ├── experiments.py        # models, CV, held-out test metrics
│   ├── decision_layer.py     # SHAP + cost-of-quality optimization
│   ├── make_figures.py       # Figures 1–2 + graphical abstract (PNG/PDF)
│   └── make_tables.py        # Tables 1–2 (CSV + LaTeX)
├── data/                     # generated dataset (lab_operations.csv)
├── results/
│   ├── figures/              # publication figures + graphical abstract
│   ├── tables/               # results tables (CSV + .tex)
│   └── metrics/              # raw metrics, SHAP values, cost analysis
├── paper/
│   ├── manuscript.tex        # IEEE (IEEEtran) source — compiles on Overleaf
│   └── manuscript.docx       # Word version (same content)
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

## Reproducing everything

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Generate the calibrated dataset
python src/generate_data.py --n 20000

# 3. Train and evaluate all models (writes metrics to results/metrics/)
python src/experiments.py

# 4. SHAP explainability + cost-of-quality decision analysis
python src/decision_layer.py

# 5. Build figures, graphical abstract, and tables
python src/make_figures.py
python src/make_tables.py
```

Run order matters: steps 3–5 consume the outputs of the previous steps. The random
seed is fixed (42) throughout, so results are deterministic.

## Data note

The dataset is **synthetic and literature-calibrated**, not real patient or donor
data. Marginal distributions and effect sizes are grounded in published
laboratory-medicine studies (see the docstring in `src/generate_data.py` and the
manuscript's data section). This makes the work fully reproducible and free of
privacy constraints; absolute figures should be read as illustrating the
methodology rather than as estimates for any specific laboratory.

## Citation

If you use this code, please cite the accompanying manuscript (see `paper/`).

## License

MIT — see `LICENSE`.
