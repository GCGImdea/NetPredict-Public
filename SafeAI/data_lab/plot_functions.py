import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_bic_and_clusters(bic_ite: np.ndarray, 
                          iterations: int, 
                          max_num_clusters: int, 
                          k_opt: int, 
                          x: np.ndarray, 
                          clusters: np.ndarray) -> None:
    meanbic = np.mean(bic_ite, axis=0)
    minbic = np.min(meanbic)
    maxbic = np.max(meanbic)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for ii in range(iterations):
        if (ii ==0):
            ax[0].plot(np.arange(1, max_num_clusters), bic_ite[ii,:], lw=2, c='lightblue', label='Realization')
        else:
            ax[0].plot(np.arange(1, max_num_clusters), bic_ite[ii,:], lw=2, c='lightblue')
    ax[0].plot(np.arange(1, max_num_clusters), meanbic, '-o',lw=2, c='darkblue', label='Average')
    ax[0].plot([k_opt, k_opt], [minbic, maxbic],'k--', lw=2)
    ax[0].set_xlabel('Number of clusters', fontsize=14)
    ax[0].set_ylabel('BIC', fontsize=14)
    # ax[0].set_title('(a)', fontsize=16)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].set_ylim([minbic-20, maxbic])
    ax[0].set_xlim([0, 40])
    # plt.text(num_opt_clusters_bic + 2, np.mean(bic_ite, axis=0)[num_opt_clusters_bic-1]-20, str(num_opt_clusters_bic) + ' clusters', fontsize=14)
    ax[0].legend(fontsize=12, loc='lower right')


    for i in range(len(np.unique(clusters))):
        ax[1].scatter(x[clusters == i], np.zeros(len(x[clusters == i])), s=100)
    ax[1].set_xlabel('Correlation Factors', fontsize=14)
    # ax[1].title('(b)', fontsize=16)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    fig.tight_layout()
    # # plt.savefig('data/figures/MonteCarloSim.png', dpi=200, bbox_inches='tight' )
    plt.show()
    



def plot_feature_scatter(out_num: pd.DataFrame, target: str, cols: pd.Index, ncols: int = 5) -> None:
    if (len(cols)%ncols) == 0 :
        nrows = len(cols)//ncols
    else:
        nrows = len(cols)//ncols + 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.00*ncols,2.50*nrows), tight_layout=True)
    for ii in range(len(cols)):
        ax[ii//ncols][ii%ncols].plot(out_num[cols[ii]], out_num[target],'o', color='tab:red')
        ax[ii//ncols][ii%ncols].set_xlabel(cols[ii], fontsize=18)
        ax[ii//ncols][ii%ncols].set_ylabel('KPI (kbps)', fontsize=18)
    fig.tight_layout()
    plt.show()