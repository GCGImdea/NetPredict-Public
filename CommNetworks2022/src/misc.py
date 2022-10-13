#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:01:26 2022

@author: juan
"""
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def feature_selection(projected_samples, clusters, centroids, corr_matrix, kpi_col):
    classes      = np.unique(clusters)
    closest      = [pairwise_distances_argmin_min(centroids[:][int(n_class)].reshape(1, -1), projected_samples[np.where(clusters == n_class)])[0] for n_class in classes] 
    corr_columns = np.array(corr_matrix.columns.values)
    train_cols   = [corr_columns[np.where(clusters == n_class)][int(closest[index])] for index, n_class in enumerate(classes)]
    train_cols   = np.delete(train_cols, clusters[corr_columns==kpi_col])
    return train_cols