#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:07:53 2022

@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_cluster2d(data, clusters, centroids, covariances, cov_type):    
    classes  = np.unique(clusters)
    fig      = plt.figure(dpi=100) 
    ax       = fig.add_subplot()
    markers  = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^","o","p"]
    
    for i in range(classes.shape[0]):
        plt.scatter(data[clusters == classes[i],0],
                    data[clusters == classes[i],1],
                    s = 75)
                #marker = markers[clusters])

    plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='x', s = 50)
    col = []
    for index in range(classes.shape[0]):
        col.append(list(plt.cm.tab10(index)))

    for uu in range(classes.shape[0]):
        if cov_type == "full":
            cov = covariances[uu][:2, :2]
        elif cov_type == "tied":
            cov = covariances[:2, :2]
        elif cov_type == "diag":
            cov = np.diag(covariances[uu][:2])
        elif cov_type == "spherical":
            cov = np.eye(centroids.shape[1]) * covariances[uu]
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(centroids[uu, :2], 
                              v[0],
                              v[1], 
                              180 + angle, 
                              edgecolor='black', 
                              facecolor='white',
                              ls ='--', 
                              lw=2)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.35)
        ax.add_artist(ell)

    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width


def hist_discretization(val, count, labels):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot()

    plt.bar(x=val, height=count, color='tab:blue', edgecolor='k', linewidth=1)
    plt.xlabel("Session time classes", fontsize=20)
    plt.ylabel("Counts", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, (6/5) * np.max(count))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
    for ii in range(len(val)):
        plt.text(val[ii],count[ii]+(0.15/5) * np.max(count),labels[ii],horizontalalignment='center', fontsize=14)
        
def plot_pruning(ccp_alphas, scores):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot()
    plt.plot(ccp_alphas, scores, marker='o', drawstyle="steps-post", color='tab:blue',linewidth=2)
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
        

def hist_differences(val, count):
    fig = plt.figure(dpi=100) 
    ax = fig.add_subplot()
    plt.bar(x=val, height=count, color='tab:blue', edgecolor='k', linewidth=1)
    plt.ylabel("Counts", fontsize=20)
    plt.xlabel("Differences", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
        
def barplot_miscoding(mscd, n_features):
    fig = plt.figure(dpi=100) 
    ax = fig.add_subplot()
    plt.bar(x=np.arange(0, len(np.argsort(-mscd)[0:n_features])), height=mscd[np.argsort(-mscd)[0:n_features]], color='tab:blue', edgecolor='k', linewidth=1, width=0.6)
    plt.xlabel("Features", fontsize=20)
    plt.ylabel("Miscoding", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width