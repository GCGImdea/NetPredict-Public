from __future__ import annotations

import numpy as np
import pandas as pd

from safeaipackage.core import rga
from safeaipackage.check_explainability import compute_rge_values
from safeaipackage.check_robustness import compute_rgr_values


def compute_rga(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Rank Graduation Accuracy (RGA) for the normality model.

    RGA measures concordance between ranks of actual and predicted values.
    Range [0, 1] -- higher is better. Unlike RMSE, robust to outliers.

    Parameters
    ----------
    y_true : actual KPI values (log-transformed)
    y_pred : predicted KPI values from the normality model

    Returns
    -------
    float : RGA score
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(rga(y_true_arr, y_pred_arr))


def compute_rge(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    model,
    variables: list[str] | None = None,
    group: bool = False,
) -> pd.DataFrame:
    """Rank Graduation Explainability (RGE) per feature.

    RGE measures the contribution of each variable to the model by comparing
    predicted ranks with and without the variable.
    RGE = 1: high contribution. RGE = 0: no contribution.

    Complements SHAP: SHAP measures marginal contribution to individual
    predictions; RGE measures rank-based contribution to overall model behaviour.

    Parameters
    ----------
    X_train   : training feature matrix
    X_test    : test/full feature matrix
    y_pred    : predictions on X_test
    model     : fitted XGBRegressor or sklearn-compatible regressor
    variables : feature names to evaluate (default: all columns of X_train)
    group     : if True, compute RGE for all variables as a group

    Returns
    -------
    pd.DataFrame with columns [feature, rge], sorted descending
    """
    if variables is None:
        variables = list(X_train.columns)

    # safeaipackage always returns a DataFrame:
    #   index = feature names (or group label), column = "RGE"
    result = compute_rge_values(
        xtrain=X_train,
        xtest=X_test,
        yhat=y_pred,
        model=model,
        variables=variables,
        group=group,
    )

    df = result.reset_index()
    df.columns = ["feature", "rge"]
    return df.sort_values("rge", ascending=False).reset_index(drop=True)


def compute_rgr(
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    model,
    variables: list[str] | None = None,
    perturbation_percentage: float = 0.05,
    group: bool = False,
) -> pd.DataFrame:
    """Rank Graduation Robustness (RGR) per feature.

    RGR measures model robustness to perturbations of each variable.
    RGR = 1: model completely robust to that variable's perturbation.
    RGR = 0: model highly sensitive.

    Parameters
    ----------
    X_test                  : test/full feature matrix
    y_pred                  : predictions on X_test
    model                   : fitted model
    variables               : feature names to evaluate (default: all)
    perturbation_percentage : fraction used as perturbation (default: 0.05)
    group                   : if True, compute RGR for all variables as a group

    Returns
    -------
    pd.DataFrame with columns [feature, rgr], sorted descending
    """
    if variables is None:
        variables = list(X_test.columns)

    # safeaipackage always returns a DataFrame:
    #   index = feature names (or group label), column = "RGR"
    result = compute_rgr_values(
        xtest=X_test,
        yhat=y_pred,
        model=model,
        variables=variables,
        perturbation_percentage=perturbation_percentage,
        group=group,
    )

    df = result.reset_index()
    df.columns = ["feature", "rgr"]
    return df.sort_values("rgr", ascending=False).reset_index(drop=True)


def safeai_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
    top_n: int = 10,
    perturbation_percentage: float = 0.05,
) -> dict:
    """Run the full SAFE AI evaluation and return a summary dict.

    Computes RGA (accuracy), RGE (explainability), and RGR (robustness)
    for the normality model.

    Parameters
    ----------
    y_true                  : actual KPI values
    y_pred                  : predicted KPI values
    X_train                 : training feature matrix
    X_test                  : full feature matrix
    model                   : fitted normality model
    top_n                   : number of top features to show in tables
    perturbation_percentage : noise fraction for RGR

    Returns
    -------
    dict with keys: rga (float), rge (DataFrame), rgr (DataFrame)
    """
    print("Computing RGA ...")
    rga_score = compute_rga(y_true, y_pred)
    print(f"  RGA = {rga_score:.4f}")

    print("Computing RGE ...")
    rge_df = compute_rge(X_train, X_test, y_pred, model)
    print(rge_df.head(top_n).to_string(index=False))

    print("Computing RGR ...")
    rgr_df = compute_rgr(
        X_test, y_pred, model,
        perturbation_percentage=perturbation_percentage,
    )
    print(rgr_df.head(top_n).to_string(index=False))

    return {
        "rga": rga_score,
        "rge": rge_df,
        "rgr": rgr_df,
    }
