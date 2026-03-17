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

from src.config import RANDOM_STATE


def mad_outliers(X: pd.Series, threshold: float = 3.0):
    Xn = X/X.std(ddof=0)
    Q3 = Xn.quantile(0.75) - Xn.mean()
    median = X.median()
    mad = (1/Q3) * (X - median).abs().median()

    if mad == 0:
        return pd.Series([False]   * len(X), index=X.index)
    else:
        return (X - median).abs() >= threshold * mad 


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