# ROAD — Interpretable Outlier and Anomaly Detection for Mobile Networks

Implementation of the ROAD methodology for unsupervised detection and interpretation
of performance outliers and anomalies in mobile network drive-test data, extended with
a SAFE AI evaluation module.

Based on:
> Ramirez et al. *Interpretable Outlier and Anomaly Detection for Mobile Networks
> from Small Tabular Data*, 2024.

SAFE AI evaluation based on:
> Babaei, Giudici & Raffinetti. *A Rank Graduation Box for SAFE AI*,
> Expert Systems with Applications, 2024.
> Giudici & Raffinetti. *RGA: a unified measure of predictive accuracy*,
> Advances in Data Analysis and Classification, 2024.

---

## Project structure

```
data_lab/
│
├── data/
│   └── aggregated_stats.csv          # raw drive-test dataset
│
├── notebooks/
│   ├── 01_preprocessing.ipynb        # data cleaning, VIF pruning, correlation pruning
│   ├── 02_anomaly_detection.ipynb    # full ROAD pipeline + SAFE AI evaluation
│   └── 03_safeai_report.ipynb        # SAFE AI results report
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # shared constants and paths
│   ├── preprocessing_functions.py    # norm_cols, vif_prune, bic_grid, ...
│   ├── detection.py                  # mad_outliers, benchmark_detectors, score_table
│   ├── explainability.py             # fit_normality_model, compute_shap_values, ...
│   ├── tree_utils.py                 # DTClassifier, build_decision_tree
│   ├── road.py                       # cluster_anomaly_support_optimal, GMMClustering1D
│   ├── plot_functions.py             # all plotting functions
│   ├── safeai.py                     # compute_rga, compute_rge, compute_rgr, safeai_summary
│   ├── run_preprocessing.py          # CLI entry point for preprocessing
│   └── run_analysis.py               # CLI entry point for detection + explainability
│
├── output/
│   ├── output_datasets/
│   │   ├── data_clean1.csv           # after basic cleaning
│   │   ├── data_clean2.csv           # after VIF pruning
│   │   ├── data_clean3.csv           # after correlation pruning (pipeline input)
│   │   └── results.pkl               # serialised results for notebook 03
│   ├── decision_trees/               # exported decision tree PNG files
│   └── figures/                      # exported plot PNG files
│
├── pyproject.toml                    # package definition for editable install
└── requirements.txt
```

---



## Methodology overview

The pipeline runs independently for each of six throughput limitations:
`1000kbit`, `2000kbit`, `5000kbit`, `10000kbit`, `20000kbit`, `50000kbit`.

Each limitation corresponds to an artificially imposed throughput cap during drive
testing, which provides ground-truth labels for evaluation.

### 1. Data preprocessing (`01_preprocessing.ipynb`)

Three sequential steps reduce the feature space from ~189 to ~120 columns:

| Step | Method | Purpose |
|---|---|---|
| Data cleaning | Missing value threshold (1%), constant column removal | Remove uninformative features |
| VIF pruning | Variance Inflation Factor, iterative removal (threshold: 20) | Remove multicollinear features |
| KPI-correlation pruning | Monte Carlo GMM clustering on Pearson correlation vector | Remove features highly correlated with target |

**Outputs:** `data_clean1.csv`, `data_clean2.csv`, `data_clean3.csv`

### 2. Outlier detection (`02_anomaly_detection.ipynb` — Section 3)

Outliers are defined as samples where the log-transformed KPI falls significantly
outside the normal distribution. Detection uses the **MAD (Median Absolute Deviation)**
method from the ROAD paper:

$$\text{MAD} = \frac{1}{Q_3} \cdot \text{median}(|y - \text{median}(y)|)$$

MAD is benchmarked against 12 state-of-the-art unsupervised detectors from PyOD
(IForest, KNN, LOF, PCA, GMM, KDE, CBLOF, COF, HBOS, COPOD, ECOD, LODA).

### 3. Anomaly detection (`02_anomaly_detection.ipynb` — Section 4)

Anomalies are samples whose KPI is within the normal range but significantly below
what a machine learning model would predict given the network conditions.

The detection pipeline has three steps:

1. **Normality model** — XGBoost regressor trained on non-outlier samples to learn
   the expected relationship between network features and throughput.
2. **Residual computation** — difference between predicted and actual log-throughput.
3. **OneClass SVM** — polynomial kernel (degree 3) fitted on residuals to identify
   samples whose residual is anomalous.

### 4. Explainability (`02_anomaly_detection.ipynb` — Sections 5–7)

Two complementary explainability methods are applied:

**SHAP (SHapley Additive exPlanations)** — measures the marginal contribution of
each feature to individual predictions of the normality model. Implemented via
`shap.TreeExplainer`.

**ROAD interpretability module** — for each feature, fits 1-D GMMs with increasing
number of clusters and computes the Jaccard index between each cluster and the anomaly
mask. The ROAD index is the maximum Jaccard score — higher means stronger association
with anomalous behaviour.

A pruned CART decision tree is then trained on the top ROAD-ranked features to
produce human-readable rules describing anomalous scenarios.

### 5. SAFE AI evaluation (`02_anomaly_detection.ipynb` — Section 6, `03_safeai_report.ipynb`)

The normality model is evaluated using three rank-based metrics from the SAFE AI
framework:

| Metric | Description | Range |
|---|---|---|
| **RGA** | Rank Graduation Accuracy — concordance between predicted and actual ranks | [0, 1] |
| **RGE** | Rank Graduation Explainability — feature contribution to rank ordering | [0, 1] |
| **RGR** | Rank Graduation Robustness — model sensitivity to feature perturbations | [0, 1] |

All three metrics operate on ranks rather than absolute values, making them robust
to the outliers that are structurally present in drive-test data.

---

## Installation

### Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

Key dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `pyod`,
`safeaipackage`, `matplotlib`, `seaborn`, `graphviz`

### Setup

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd data_lab

python -m venv .venv
source .venv/bin/activate
````

Install the project in editable mode:

```bash
pip install --upgrade pip
pip install -e .
```

(Optional) Install development dependencies (e.g., for notebooks):

```bash
pip install -e ".[dev]"
```

---

## Usage

### Run the full pipeline from notebooks

Open Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

1. `notebooks/01_preprocessing.ipynb` — produces `output/output_datasets/data_clean*.csv`
2. `notebooks/02_anomaly_detection.ipynb` — runs detection, explainability and SAFE AI
3. `notebooks/03_safeai_report.ipynb` — loads saved results and produces the report

At the end of notebook 02, save the results to disk for notebook 03:

### Run preprocessing from the command line

```bash
python -m src.run_preprocessing
```

This produces the three cleaned CSV files in `output_datasets/` without opening Jupyter.

#TODO test the function

---

## Configuration

All shared constants are defined in `src/config.py`:

```python
TARGET       = "dl.throughput.value"   # KPI column name
MISSING_MAX  = 0.01                    # max fraction of missing values per column
VIF_MAX      = 20.0                    # VIF threshold for multicollinearity pruning
BIC_ITERATIONS = 50                    # Monte Carlo iterations for GMM BIC estimation
MAD_THRESHOLD  = 3.0                   # threshold multiplier for MAD outlier detection
DT_MAX_DEPTH   = 10                    # decision tree maximum depth
RANDOM_STATE   = 42
```

To change any parameter, edit `src/config.py` — changes propagate automatically
to all notebooks and scripts.

#TODO Fix the output dataset for the preprocessing 



## References

```
Ramirez, J.M., Rojo, P., Mancuso, V., & Fernandez-Anta, A. (2024).
  Interpretable Outlier and Anomaly Detection for Mobile Networks
  from Small Tabular Data.

Babaei, G., Giudici, P., & Raffinetti, E. (2024).
  A Rank Graduation Box for SAFE AI.
  Expert Systems with Applications, 125239.

Giudici, P., & Raffinetti, E. (2024).
  RGA: a unified measure of predictive accuracy.
  Advances in Data Analysis and Classification, 19, 67–93.

Raffinetti, E. (2023).
  A rank graduation accuracy measure to mitigate Artificial Intelligence risks.
  Quality & Quantity, 57, S131–S150.
```