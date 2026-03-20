import numpy as np
import pandas as pd
import xgboost as xgb


from data_lab.config import RANDOM_STATE, TARGET

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
    Shape returned: (n_samples,)
    """
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = y_pred_arr - y_true_arr
    return residuals


def detect_residual_anomalies(
    residuals: np.ndarray,
    threshold: float = 3.5,
    use_absolute: bool = True,
) -> np.ndarray:
    """
    Detect anomalies from residuals using a robust MAD-based threshold.
    Returns a boolean array: True = anomaly.
    """
    r = np.asarray(residuals, dtype=float).reshape(-1)

    if use_absolute:
        r = np.abs(r)

    median = np.median(r)
    mad = np.median(np.abs(r - median))

    if mad == 0:
        return r > median

    robust_z = 0.6745 * (r - median) / mad
    return robust_z > threshold


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
