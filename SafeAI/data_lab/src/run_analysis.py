from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_PATH, TARGET, THROUGHPUT_LIMS
from src.detection import mad_outliers, benchmark_detectors, score_table
from src.explainability import (
    prepare_training_split,
    fit_normality_model,
    predict_target,
    compute_residuals,
    detect_residual_anomalies,
    compute_shap_values,
    rank_features_by_mean_abs_shap,
)


def load_prepared_data(path: str | None = None) -> pd.DataFrame:
    data_path = path or DATA_PATH
    return pd.read_csv(data_path, encoding="ISO-8859-1", low_memory=False)


def build_limit_subset(data_prepared: pd.DataFrame, lim: str, target: str = TARGET):
    lim_mask = data_prepared["limitation"].isna() | (data_prepared["throughputlim"] == lim)
    data_lim = data_prepared.loc[lim_mask].copy()

    outliers_real = (data_lim["throughputlim"] == lim).astype(int)

    positive_mask = data_lim[target] > 0
    data_lim = data_lim.loc[positive_mask].copy()
    outliers_real = outliers_real.loc[data_lim.index]

    X_lim = data_lim.select_dtypes(include="number").copy()
    X_lim[target] = np.log10(X_lim[target])

    return data_lim, X_lim, outliers_real


def run_single_limit(data_prepared: pd.DataFrame, lim: str):
    data_lim, X_lim, outliers_real = build_limit_subset(data_prepared, lim)

    outliers_mad = mad_outliers(X_lim[TARGET])
    contamination = float(outliers_real.mean())

    comparison = benchmark_detectors(X_lim, outliers_real, contamination)
    comparison.loc[len(comparison)] = {
        "method": "MAD",
        **score_table(outliers_real, outliers_mad.astype(int)),
    }
    comparison = comparison.sort_values("f1_score", ascending=False).reset_index(drop=True)

    X_train, y_train, X_all = prepare_training_split(X_lim, outliers_mad, target=TARGET)

    model_normality = fit_normality_model(X_train, y_train)
    y_predicted = predict_target(model_normality, X_all)
    residuals = compute_residuals(X_lim[TARGET], y_predicted)
    residual_anomalies = detect_residual_anomalies(residuals)

    shap_values = compute_shap_values(model_normality, X_train)
    shap_ranking = rank_features_by_mean_abs_shap(shap_values, X_train.columns)

    return {
        "limit": lim,
        "data_lim": data_lim,
        "X_lim": X_lim,
        "outliers_real": outliers_real,
        "outliers_mad": outliers_mad,
        "comparison": comparison,
        "X_train": X_train,
        "y_train": y_train,
        "model_normality": model_normality,
        "y_predicted": y_predicted,
        "residuals": residuals,
        "residual_anomalies": residual_anomalies,
        "shap_values": shap_values,
        "shap_ranking": shap_ranking,
    }


def main():
    data_prepared = load_prepared_data()

    all_results = {}
    for lim in THROUGHPUT_LIMS:
        result = run_single_limit(data_prepared, lim)
        all_results[lim] = result

        print(f"\n=== {lim} ===")
        print(result["comparison"].to_string(index=False))
        print("\nTop SHAP features:")
        print(result["shap_ranking"].head(10).to_string(index=False))
        print(f"\nResidual anomalies: {result['residual_anomalies'].sum()} / {len(result['residual_anomalies'])}")

    return all_results


if __name__ == "__main__":
    main()