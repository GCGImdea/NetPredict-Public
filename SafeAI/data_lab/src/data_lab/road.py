from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn import mixture
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler


class GMMClustering1D:
    """1-D Gaussian Mixture Model with BIC-based model selection.

    Fits GMMs with n_components from 2 to n_max and keeps the best one.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def fit(self, n_max: int, rnd_state: int = 1234, cov_type: str = "spherical") -> GMMClustering1D:
        lowest_bic = np.inf
        best_gmm   = None
        self._all_clusters: list[np.ndarray] = []
        self._all_classes:  list[np.ndarray] = []
        self.n_degenerate = 0

        for n in range(2, n_max + 1):
            gmm = mixture.GaussianMixture(
                n_components=n,
                covariance_type=cov_type,
                random_state=rnd_state,
            )
            gmm.fit(self.data)
            labels = gmm.predict(self.data)
            self._all_clusters.append(labels)
            self._all_classes.append(np.unique(labels))

            if len(np.unique(labels)) <= 1:
                self.n_degenerate += 1

            bic = gmm.bic(self.data)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm   = gmm

        self.clusters_    = best_gmm.predict(self.data)
        self.centroids_   = best_gmm.means_
        self.covariances_ = best_gmm.covariances_
        return self


def cluster_anomaly_support_optimal(
    data: pd.DataFrame,
    yanm: pd.Series,
    attributes: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rank features by association with anomalies via 1-D GMM clustering (ROAD index).

    For each feature, fits GMMs with increasing number of clusters and computes
    the Jaccard index between each cluster and the anomaly mask. The ROAD index
    for a feature is the maximum Jaccard score across all clusters and all n_components.

    Parameters
    ----------
    data       : feature matrix (numeric)
    yanm       : boolean anomaly mask (True = anomaly)
    attributes : feature names to analyse
    cat_cols   : categorical columns — skip GMM, use values directly

    Returns
    -------
    models  : DataFrame with columns [Attribute, best_model, Class_1, Class_2],
              sorted by ROAD index descending
    cls_mat : class assignment matrix (n_samples × n_top_features)
    """
    n_max_clusters = int(np.ceil(np.log2(np.log2(len(yanm)))))

    all_ratios: list[float]      = []
    all_featrs: list[str]        = []
    all_pattrn: list[np.ndarray] = []

    for attr_name in attributes:
        ni = 0
        attr = data[attr_name]

        if attr_name in cat_cols:
            clustering   = attr.values
            cluster_app  = [clustering]
            classes_app  = [np.unique(clustering)]
        else:
            scaler   = StandardScaler()
            data2prc = scaler.fit_transform(attr.values.reshape(-1, 1))

            model = GMMClustering1D(data2prc)
            model.fit(n_max=8)
            cluster_app = model._all_clusters
            classes_app = model._all_classes
            ni          = model.n_degenerate

        ratio:   np.ndarray  = np.array([])
        pattern: list        = []

        if ni != 8:
            for kk in range(len(cluster_app)):
                if len(classes_app[kk]) > 1:
                    for ii in range(len(classes_app[kk])):
                        score = jaccard_score(yanm, (cluster_app[kk] == ii))
                        ratio = np.append(ratio, score)
                        pattern.append(cluster_app[kk] == ii)

        if len(ratio) > 0:
            all_ratios.append(float(np.max(ratio)))
            all_featrs.append(attr_name)
            all_pattrn.append(pattern[int(np.argmax(ratio))])

    # sort by ROAD index descending
    sort_idx   = np.argsort(-np.array(all_ratios))
    all_ratios = [all_ratios[i] for i in sort_idx]
    all_featrs = [all_featrs[i] for i in sort_idx]
    all_pattrn = [all_pattrn[i] for i in sort_idx]

    # build class matrix for the top n_prob features
    n_prob        = int(np.ceil(np.log2(np.log2(len(yanm)))))
    cls_mat       = np.zeros((len(yanm), n_prob))
    class_problem = np.zeros(len(yanm))

    rows = []
    for i in range(min(n_prob, len(all_featrs))):
        class_problem[all_pattrn[i]]  = 1
        class_problem[~all_pattrn[i]] = 2
        cls_mat[:, i] = class_problem

        counts = np.array([
            int((class_problem == 1).sum()),
            int((class_problem == 2).sum()),
        ])
        total = counts.sum()
        rows.append({
            "Attribute":  all_featrs[i],
            "road_index": all_ratios[i],
            "Class_1":    counts[0] / total,
            "Class_2":    counts[1] / total,
        })

    models = pd.DataFrame(rows)
    return models, cls_mat
