#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import mixture

# Scikit-learn libraries (models)
from sklearn.tree import DecisionTreeClassifier

# Scikit-learn libraries (cross-validation)
from sklearn.model_selection import cross_val_score



class GMMclustering():
    def __init__(
            self, 
            data):
        self.data = data

    def predict(self, n_max, rnd_state, cov_type):
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, n_max+1)
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=rnd_state)
            gmm.fit(self.data)
            bic.append(gmm.bic(self.data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
        self.bic_ = np.array(bic)
        clf = best_gmm
        self.clusters_ = clf.predict(self.data)
        self.centroids_ = clf.means_
        self.covariances_ = clf.covariances_
        return self 


def model_based_clustering(input_data, n_max, rnd_state, covariance):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_max+1)
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance, random_state=rnd_state)
        gmm.fit(input_data)
        bic.append(gmm.bic(input_data))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    bic = np.array(bic)
    clf = best_gmm
    clustering = clf.predict(input_data)
    centroids = clf.means_
    covariances = clf.covariances_
    return clustering, centroids, covariances, bic


def build_decision_tree(depth, min_samples, X_train, y_train, train_split):
    model = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = min_samples)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        aux_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples, ccp_alpha=ccp_alpha)
        aux_model.fit(X_train, y_train)
        clfs.append(aux_model)
    scores = [cross_val_score(aux_model, X_train, y_train, cv=train_split, n_jobs=-1).mean() for aux_model in clfs]
    best_alpha = ccp_alphas[np.where(scores == np.max(scores))[0][0]]
    len(ccp_alphas), np.where(scores == np.max(scores))[0][0], best_alpha
    
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples, ccp_alpha=best_alpha)
    model.fit(X_train, y_train)
    return model, scores, ccp_alphas, best_alpha