from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(
    model,
    X_background: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for a tree-based regressor.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    return shap_values


def rank_features_by_mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """
    Rank features by average absolute SHAP importance.
    """
    importance = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return ranking.reset_index(drop=True)


