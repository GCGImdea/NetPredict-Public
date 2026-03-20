from __future__ import annotations

import time
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from data_lab.config import RANDOM_STATE


def build_decision_tree(max_depth, min_samples_leaf, X_train, y_train, cv_folds):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    clfs = []
    scores = []

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=alpha,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        score = cross_val_score(clf, X_train, y_train, cv=cv_folds, n_jobs=-1).mean()
        clfs.append(clf)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_alpha = ccp_alphas[best_idx]
    best_model = clfs[best_idx]

    return best_model, scores, ccp_alphas, best_alpha


class DTClassifier:
    def __init__(self, max_depth, min_samples_leaf, cv_folds):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cv_folds = cv_folds

    def fit(self, X, y):
        start_time = time.time()
        self.model, self.scores, self.alphas, self.best_alpha = build_decision_tree(
            self.max_depth,
            self.min_samples_leaf,
            X,
            y,
            self.cv_folds,
        )
        print(f"Model training finished in {time.time() - start_time:.4f} seconds")
        return self

    def predict(self, X):
        return self.model.predict(X)