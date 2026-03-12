#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import PyOD models
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.sod import SOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA
from pyod.models.deep_svdd import DeepSVDD

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import OneClassSVM

from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

# import statsmodels.api as sm
# import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


data_files = ['d2_Operator1_CapacityDL_clean.csv',
              'd2_Operator1_CapacityUL_clean.csv',
              'd2_Operator1_HttpTransferDL_clean.csv',
              'd2_Operator1_HttpTransferUL_clean.csv',
              'd1_HttpFileDL_num_clean.csv',
              'd1_HttpFileUL_num_clean.csv']

paper_subsets = ['1A',
                 '1B',
                 '1C',
                 '1D',
                 '2A',
                 '2B']


comparison = pd.DataFrame(columns=['subset',
                                   'method',
                                   'sensitivity',
                                   'specificity',
                                   'f1-score',
                                   'jaccard',
                                   'aucroc'])

for i in range(len(data_files)):
    data = pd.read_csv('data/clean_data/' + data_files[i] , sep=",", decimal='.', low_memory = False)
    print('Data Size:(%d, %d)'%(data.shape[0], data.shape[1]))
    print('-----------------------------------------')
    target_kpi = 'transfer.datarate'
    kpi_aux    = data[target_kpi]


    data = data.loc[:,data.columns != target_kpi]
    data[target_kpi] = np.log10(kpi_aux)

    data_nrm = data[target_kpi].to_numpy() / np.std(data[target_kpi])
    Q3 = np.quantile(data_nrm, 0.75) - np.mean(data_nrm)
    if (Q3 != 0):
        med = np.median(data[target_kpi])
        MAD = (1/Q3) * np.median(np.absolute(data[target_kpi] - med))
        outliers_kpi = (data[target_kpi] > med + 3.0*MAD)|(data[target_kpi] < med - 3.0*MAD)
    print('Outlier Rate in KPI: %.4f'%(np.mean(outliers_kpi)))
    
    
    anomaly_classes = np.unique(outliers_kpi)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.hist([data[target_kpi].to_numpy()[outliers_kpi == i] for i in anomaly_classes], 
            bins=30, 
            density = False,
            color=[sns.color_palette("tab20")[1],sns.color_palette("tab20")[0]],
            stacked=True,
            label = ['non outliers', 
                     'outliers'],
            edgecolor = 'w')
    plt.tick_params(axis='both', labelsize=22)
    plt.legend(fontsize=22)
    plt.xlabel('log10(transfer.datarate)', fontsize=26)
    plt.ylabel('Count', fontsize=26)
    plt.title('Subset ' + paper_subsets[i],fontsize=26)
    plt.autoscale(axis='both', tight=True)
    plt.savefig('data/figures/OD_hist_Subset_' + paper_subsets[i] +'.png', bbox_inches='tight' )
    
    
    outlier_rate = np.mean(outliers_kpi)
    random_state = 42
    classifiers ={'IForest': IForest(contamination = outlier_rate,
                                              bootstrap = True,
                                              random_state = random_state),
                  'KNN':      KNN(contamination=outlier_rate),
    	              'LOF':      LOF(n_neighbors=35, 
                                  contamination=outlier_rate),
                  'PCA':      PCA(contamination=outlier_rate, 
                                  random_state=random_state),
    	              'GMM':      GMM(contamination=outlier_rate,
    							  random_state=random_state),
    	              'KDE':      KDE(contamination=outlier_rate),
                  'OCSVM':    OCSVM(contamination=outlier_rate,
                                    kernel='poly',
                                    degree=3),
                  'CBLOF':    CBLOF(contamination=outlier_rate,
    			                    check_estimator=False, 
                                    random_state=random_state),
                  'COF':      COF(contamination=outlier_rate,
                                  n_neighbors=35,),
                  'HBOS':     HBOS(n_bins='auto',
                                   contamination=outlier_rate),
                  'COPOD':    COPOD(contamination=outlier_rate),
                  'ECOD':     ECOD(contamination=outlier_rate),
                  'LODA':     LODA(contamination=outlier_rate),
    }
    

    X = data.iloc[:, data.columns == target_kpi]
    for j, (clf_name, clf) in enumerate(classifiers.items()):
        print(j + 1, 'fitting', clf_name)
        clf.fit(X)
        y_pred  = clf.predict(X)
        # y_score = clf.predict_score(X)
        comparison.at[len(classifiers.items())*i + j,'method'] = clf_name
        comparison.at[len(classifiers.items())*i + j,'sensitivity'] = recall_score(outliers_kpi, y_pred)
        comparison.at[len(classifiers.items())*i + j,'specificity'] = recall_score(outliers_kpi, y_pred, pos_label=0)
        comparison.at[len(classifiers.items())*i + j,'f1-score']    = f1_score(outliers_kpi, y_pred)
        comparison.at[len(classifiers.items())*i + j,'jaccard']     = jaccard_score(outliers_kpi, y_pred)
        comparison.at[len(classifiers.items())*i + j,'aucroc']      = roc_auc_score(outliers_kpi, y_pred)
        comparison.at[len(classifiers.items())*i + j,'subset']      = paper_subsets[i]


methods = comparison['method'].unique()
num_methods = len(methods)

angles = np.linspace(0, 2 * np.pi, num_methods, endpoint=False).tolist()
angles += angles[:1]  # Close the circle

# Initialize plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for subset in paper_subsets:
    subset_data = comparison[comparison['subset']==subset]['sensitivity'].values.flatten().tolist()
    subset_data += subset_data[:1]  # Close the line
    ax.plot(angles, subset_data, linewidth=3, label=f'{subset}')


# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(100)
ax.tick_params(axis='x', which='major', pad=20)
plt.xticks(angles[:-1], methods, fontsize=22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.ylim(0, 1)
plt.title('Sensitivity', fontsize=32, pad=50)
plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1.30, 1.1))
plt.show()


# Initialize plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for subset in paper_subsets:
    subset_data = comparison[comparison['subset']==subset]['specificity'].values.flatten().tolist()
    subset_data += subset_data[:1]  # Close the line
    ax.plot(angles, subset_data, linewidth=3, label=f'{subset}')


# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(100)
ax.tick_params(axis='x', which='major', pad=20)
plt.xticks(angles[:-1], methods, fontsize=22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.ylim(0, 1)
plt.title('Specificity', fontsize=32, pad=50)
plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1.30, 1.1))
plt.show()

# Initialize plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for subset in paper_subsets:
    subset_data = comparison[comparison['subset']==subset]['f1-score'].values.flatten().tolist()
    subset_data += subset_data[:1]  # Close the line
    ax.plot(angles, subset_data, linewidth=3, label=f'{subset}')


# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(100)
ax.tick_params(axis='x', which='major', pad=20)
plt.xticks(angles[:-1], methods, fontsize=22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.ylim(0, 1)
plt.title('(a) F1-Score', fontsize=32, pad=50)
plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1.30, 1.1))
fig.set_size_inches(8.,8.)
plt.savefig('data/figures/OD_radar_f1score.png', dpi=200, bbox_inches='tight' )
plt.show()



# Initialize plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for subset in paper_subsets:
    subset_data = comparison[comparison['subset']==subset]['jaccard'].values.flatten().tolist()
    subset_data += subset_data[:1]  # Close the line
    ax.plot(angles, subset_data, linewidth=3, label=f'{subset}')


# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(100)
ax.tick_params(axis='x', which='major', pad=20)
plt.xticks(angles[:-1], methods, fontsize=22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.ylim(0, 1)
plt.title('Jaccard', fontsize=32, pad=50)
plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1.30, 1.1))
plt.show()


# Initialize plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for subset in paper_subsets:
    subset_data = comparison[comparison['subset']==subset]['aucroc'].values.flatten().tolist()
    subset_data += subset_data[:1]  # Close the line
    ax.plot(angles, subset_data, linewidth=3, label=f'{subset}')


# Customize plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(100)
ax.tick_params(axis='x', which='major', pad=20)
plt.xticks(angles[:-1], methods, fontsize=22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
plt.ylim(0, 1)
plt.title('(b) AUCROC', fontsize=32, pad=50)
plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1.30, 1.1))
fig.set_size_inches(8.,8.)
plt.savefig('data/figures/OD_radar_aucroc.png', dpi=200, bbox_inches='tight' )
plt.show()