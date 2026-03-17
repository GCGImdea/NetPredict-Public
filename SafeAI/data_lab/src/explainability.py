from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sklearn.svm import OneClassSVM

from src.config import RANDOM_STATE, TARGET


def fit_normality_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBRegressor:
    """
    Fit a regressor on the non-anomalous portion of the data.
    """
    model = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    return model


def predict_target(
    model,
    X: pd.DataFrame,
    min_positive: float = 1e-3,
) -> np.ndarray:
    """
    Predict target values and clip non-positive values to avoid downstream issues.
    """
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred[y_pred <= 0] = min_positive
    return y_pred


def compute_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Residuals used by the anomaly detector.
    Shape returned: (n_samples, 1)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_pred_arr - y_true_arr
    return residuals.reshape(-1, 1)


def detect_residual_anomalies(
    residuals: np.ndarray,
    kernel: str = "poly",
    degree: int = 3,
    gamma: str = "auto",
) -> np.ndarray:
    """
    Detect anomalies from residuals using One-Class SVM.
    Returns a boolean array: True = anomaly.
    """
    detector = OneClassSVM(gamma=gamma, kernel=kernel, degree=degree)
    detector.fit(residuals)
    return detector.predict(residuals) == -1


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


def prepare_training_split(
    X_lim: pd.DataFrame,
    mad_mask: pd.Series,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build training set from non-MAD samples.
    Returns:
      - X_train
      - y_train
      - X_features_all
    """
    feature_cols = [c for c in X_lim.columns if c != target]
    X_train = X_lim.loc[~mad_mask, feature_cols]
    y_train = X_lim.loc[~mad_mask, target]
    X_all = X_lim.loc[:, feature_cols]
    return X_train, y_train, X_all