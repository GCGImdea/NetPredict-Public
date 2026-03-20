from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA

from data_lab.config import RANDOM_STATE

def mad_outliers(x: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers in a 1D series using the classical Median Absolute Deviation (MAD).

    The method is robust because it relies on:
    - the median as measure of central tendency
    - the MAD as measure of dispersion

    For each value x_i, it computes a robust z-score:

        robust_z_i = 0.6745 * |x_i - median(x)| / MAD

    where:

        MAD = median(|x - median(x)|)

    The constant 0.6745 makes the score comparable to the standard z-score
    under normality assumptions.

    Parameters
    ----------
    x : pd.Series
        Input 1D data.
    threshold : float, default=3.5
        Robust z-score threshold above which a point is flagged as outlier.
        Common choices:
        - 3.0  -> more sensitive
        - 3.5  -> standard robust default
        - 4.0+ -> more conservative

    Returns
    -------
    pd.Series
        Boolean Series aligned with x.index:
        True  -> outlier
        False -> inlier

    Notes
    -----
    - This method is symmetric around the median.
    - It works well when the distribution is roughly symmetric or when
      a simple robust baseline is needed.
    - If the distribution is strongly skewed, Double MAD may be preferable.

    Edge cases
    ----------
    If MAD == 0, the function returns all False.
    This means that no point is flagged as outlier because the data show
    no robust spread around the median.

    Example
    -------
    >>> s = pd.Series([10, 11, 10, 12, 9, 10, 100])
    >>> mad_outliers(s)
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool
    """
    x = pd.Series(x, copy=False)

    median = x.median()
    abs_dev = (x - median).abs()
    mad = abs_dev.median()

    if mad == 0 or pd.isna(mad):
        return pd.Series(False, index=x.index)

    robust_z = 0.6745 * abs_dev / mad
    return robust_z > threshold


def double_mad_outliers(x: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    based on https://aakinshin.net/posts/harrell-davis-double-mad-outlier-detector/#Rosenmai2013
    Detect outliers in a 1D series using the Double MAD approach.

    Unlike the standard MAD, Double MAD handles asymmetric distributions by
    computing two different dispersion estimates:
    - left MAD  for values <= median
    - right MAD for values >= median

    For each value x_i:
    - if x_i <= median, the left MAD is used
    - if x_i >= median, the right MAD is used

    The robust score is:

        score_i = 0.6745 * |x_i - median(x)| / MAD_side

    where MAD_side is either the left or right MAD depending on the side
    of the median.

    Parameters
    ----------
    x : pd.Series
        Input 1D data.
    threshold : float, default=3.5
        Robust score threshold above which a point is flagged as outlier.

    Returns
    -------
    pd.Series
        Boolean Series aligned with x.index:
        True  -> outlier
        False -> inlier

    Why use this method
    -------------------
    Standard MAD assumes a symmetric spread around the median.
    Double MAD is more appropriate when:
    - the data are skewed
    - one tail is naturally longer than the other
    - the same threshold should not be applied equally to both sides

    Edge cases
    ----------
    - If one side MAD is zero, points on that side are not flagged using
      a division by zero. In that case, their score is set to 0.
    - The median itself is never flagged as outlier.

    Example
    -------
    >>> s = pd.Series([1, 2, 2, 3, 3, 4, 5, 20])
    >>> double_mad_outliers(s)
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6    False
    7     True
    dtype: bool
    """
    x = pd.Series(x, copy=False)

    median = x.median()

    left_mask = x < median
    right_mask = x > median
    median_mask = x == median

    left = x[left_mask]
    right = x[right_mask]

    left_mad = (median - left).median() if len(left) > 0 else np.nan
    right_mad = (right - median).median() if len(right) > 0 else np.nan

    scores = pd.Series(0.0, index=x.index)

    if pd.notna(left_mad) and left_mad != 0:
        scores.loc[left_mask] = 0.6745 * (median - x.loc[left_mask]).abs() / left_mad

    if pd.notna(right_mad) and right_mad != 0:
        scores.loc[right_mask] = 0.6745 * (x.loc[right_mask] - median).abs() / right_mad

    scores.loc[median_mask] = 0.0

    return scores > threshold


def make_detectors(contamination: float) -> dict[str, object]:
    return {
        "IForest": IForest(contamination=contamination, bootstrap=True, random_state=RANDOM_STATE),
        "KNN": KNN(contamination=contamination),
        "LOF": LOF(n_neighbors=35, contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=RANDOM_STATE),
        "GMM": GMM(contamination=contamination, random_state=RANDOM_STATE),
        "KDE": KDE(contamination=contamination),
        "CBLOF": CBLOF(contamination=contamination, check_estimator=False, random_state=RANDOM_STATE),
        "COF": COF(contamination=contamination, n_neighbors=35),
        "HBOS": HBOS(n_bins="auto", contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "ECOD": ECOD(contamination=contamination),
        "LODA": LODA(contamination=contamination),
    }


def score_table(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    return {
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "jaccard": jaccard_score(y_true, y_pred, zero_division=0),
    }


def benchmark_detectors(X: pd.DataFrame, y_true: pd.Series, contamination: float) -> pd.DataFrame:
    rows = []

    for name, clf in make_detectors(contamination).items():
        clf.fit(X)
        y_pred = pd.Series(clf.predict(X), index=X.index)
        y_pred = y_pred.replace({-1: 1})
        rows.append({"method": name, **score_table(y_true, y_pred)})

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False)