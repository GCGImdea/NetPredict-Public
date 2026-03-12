#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time

# Ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import colors

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

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

from sklearn import mixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

# import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb

import shap
import graphviz

import warnings
warnings.filterwarnings('ignore')


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


    # model = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = min_samples)
    # path = model.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # clfs = []
    # for ccp_alpha in ccp_alphas:
    #     aux_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples, ccp_alpha=ccp_alpha)
    #     aux_model.fit(X_train, y_train)
    #     clfs.append(aux_model)
    # scores = [cross_val_score(aux_model, X_train, y_train, cv=train_split, n_jobs=-1).mean() for aux_model in clfs]
    # best_alpha = ccp_alphas[np.where(scores == np.max(scores))[0][0]]
    # len(ccp_alphas), np.where(scores == np.max(scores))[0][0], best_alpha
    
    # model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples, ccp_alpha=best_alpha)
    model.fit(X_train, y_train)
    # return model, scores, ccp_alphas, best_alpha
    return model


class DTClassifier():
    def __init__(self, n_classes, n_leafs, cross_val_param):
        self.n_classes = n_classes
        self.n_leafs = n_leafs
        self.cross_val_param = cross_val_param
        
    def fit(self, X, y):
        start_time = time.time()
        # min_counts = 0
        # nbins = self.n_classes
        # while (min_counts < 30):
        #     discretization_model = KBinsDiscretizer(n_bins= nbins, 
        #                                             encode='ordinal', 
        #                                             strategy='kmeans').fit(y)
        #     self.y_discretized = discretization_model.transform(y)[:,0]
        #     vals, counts = np.unique(self.y_discretized, return_counts=True)
        #     min_counts = np.min(counts)
        #     nbins -= 1
        
        self.model, self.scores, self.alphas, self.best_alpha = build_decision_tree(self.n_classes, 
                                                                                        self.n_leafs, 
                                                                                        X, 
                                                                                        y, 
                                                                                        self.cross_val_param)
        print('Model training Finished... Elapsed time: %.4f seconds'%(time.time() - start_time))
        return self
    def predict(self, X):
        self.predicted = self.model.predict(X)
        return self.predicted


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
            gmm = mixture.GaussianMixture(n_components=n_components, 
                                          covariance_type=cov_type, 
                                          random_state=rnd_state)
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
    


def cluster_anomaly_support_optimal(data, yanm, attributes, cat_cols, file_save):
    num_max_cls = np.ceil(np.log2(np.log2(len(yanm)))).astype(int)
    all_ratios = np.array([])
    all_featrs = np.array([])
    cluster_ratios = []
    class_ratios   = []
    all_pattrn = []
    for i in range(len(attributes)):
        attr = data[attributes[i]]
        scaler = StandardScaler()
        scaler.fit(attr.values.reshape(-1, 1))
        data2prc = scaler.transform(attr.values.reshape(-1, 1))
        
        if attributes[i] in cat_cols:
            clustering = data[attributes[i]]
        else:
            cluster1dmodel = GMMclustering(data2prc)
            
            cluster_app = []
            classes_app = []
            ni = 0
            for ll in range (2, 10):
                cluster1dmodel.predict(ll, 1234, 'spherical')
                clustering = cluster1dmodel.clusters_
                cluster_app.append(clustering)
                classes_app.append(np.unique(clustering))
                if (len(np.unique(clustering)) <= 1):
                    ni += 1
            
        classes = np.unique(clustering)
        cluster_ratios.append(clustering)
        class_ratios.append(classes)
        
        # print(classes_app, ni)
        ratio = np.array([])
        pattern = []
        if (ni != 8):
            kk=0
            for ll in range (2, 10):
                if (len(classes_app[kk]) > 1):
                    for ii in range(len(classes_app[kk])):
                        ratio = np.append(ratio, jaccard_score(yanm, (cluster_app[kk] == ii)))
                        pattern.append((cluster_app[kk] == ii))
                kk += 1
                        
            all_ratios = np.append(all_ratios, np.max(ratio))
            all_featrs = np.append(all_featrs, attributes[i])
            all_pattrn.append(pattern[np.argmax(ratio)])
            
    all_featrs = all_featrs[np.argsort(-all_ratios)]
    sort_indices = np.argsort(-all_ratios)
    
    print(all_ratios[sort_indices])
    print(all_featrs)
    
    
    num_features = 12
    if (len(attributes) >= num_features):
        fig = plt.figure(figsize=(0.9*4,0.9*3))
        plt.barh(np.arange(num_features,0,-1), 
                 all_ratios[sort_indices][0:num_features],
                 height=0.50,
                 color=sns.color_palette("tab20")[1],
                 edgecolor=sns.color_palette("tab20")[0])
        plt.yticks(np.arange(num_features,0,-1), all_featrs[0:num_features], fontsize=11)
        plt.title('(a) ROAD ranking (anomalies)', fontsize=14)
        plt.savefig('data/figures/' + file_save + '.png',dpi=150, bbox_inches='tight')
        plt.show()
    
    
    num_prob_att = np.ceil(np.log2(np.log2(len(yanm)))).astype(int)
    models = pd.DataFrame({"Attribute":pd.Series(dtype='str'),
                            "best_model":pd.Series(dtype='str'),
                            "Class_1":pd.Series(dtype='float'),
                            "Class_2":pd.Series(dtype='float')}) 
    cls_mat = np.zeros((len(yanm), num_prob_att))
    class_problem = np.zeros(len(yanm))
    for i in range(num_prob_att):
        class_problem[all_pattrn[sort_indices[i]] == True]  = 1
        class_problem[all_pattrn[sort_indices[i]] == False] = 2
        cls_mat[:,i] = class_problem       
        
        counts = np.array([class_problem[class_problem==1].shape[0], class_problem[class_problem==2].shape[0]])
        total  = np.sum(counts)
        nm_row = pd.DataFrame([[all_featrs[i],'best_model', counts[0]/total, counts[1]/total]], 
                      columns = ['Attribute', 'best_model', 'Class_1','Class_2'])
        models = pd.concat([models, nm_row], sort=False, ignore_index=True)
    return models, cls_mat




# data_file    = 'd2_Operator1_HttpTransferUL_clean.csv'
data_file    = 'd2_Operator1_CapacityDL_clean.csv'
paper_subset = '1A'




comparison = pd.DataFrame(columns=['subset',
                                   'method',
                                   'sensitivity',
                                   'specificity',
                                   'f1-score',
                                   'jaccard',
                                   'aucroc'])


regressor = 'XGB'


# datax = np.load('data/clean_data/6_cardio.npz', allow_pickle=True)
datax = np.load('data/clean_data/31_satimage-2.npz', allow_pickle=True)
# datax = np.load('data/clean_data/41_Waveform.npz', allow_pickle=True)
X, y = datax['X'], datax['y']

anm_labels = []
for ii in range(X.shape[1]):
    anm_labels.append('satimage_'+ str(ii))
    # anm_labels.append('cardio_'+ str(ii))
Xdf = pd.DataFrame(X,columns=anm_labels)

print('-----------------------------------------------------------------')
print('CLUSTER DENSITIES')
models, cls_mat = cluster_anomaly_support_optimal(Xdf, y, anm_labels, [],'satimage')
print(models)

attributes = models["Attribute"].values
attributes, att_ind = np.unique(attributes, return_index=True)
cls_mat = cls_mat[:,att_ind]
problem = np.empty(cls_mat.shape[0], dtype=object)
separator = ';'
for ii in range(cls_mat.shape[0]):
    print(ii)
    strings = []
    for jj in range(cls_mat.shape[1]):
        if ((cls_mat[ii,jj] == 1)&(y[ii] == True)):
#         if (cls_mat[ii,jj] == 1):
            strings = strings + ['T%dP'%(jj+1)]
        if (cls_mat[ii,jj] == 2)|((cls_mat[ii,jj] == 1)&(y[ii] == False)):
#         if (cls_mat[ii,jj] == 2):
            strings = strings + ['T%dN'%(jj+1)]
    problem[ii] = separator.join(strings)
    
k = 0;
# lbx = ['(i)', '(g)', '(h)', '(j)']
lbx = ['(c)', '(e)', '(d)', '(f)']
for attr in attributes:
    cls_vec = cls_mat[:,k]
    bl_vec1 = ((cls_vec==1)&(y==True))
    bl_vec2 = (cls_vec==2)|((cls_vec==1)&(y==False))
#     bl_vec1 = (cls_vec==1)
#     bl_vec2 = (cls_vec==2)
    fig = plt.figure(dpi=150,figsize=(4,3)) #Paper figure
    ax = fig.add_subplot()
    plt.scatter(Xdf[attr][bl_vec2], np.zeros(Xdf[attr][bl_vec2].shape),label='T%dN'%(k+1), marker='o',color=sns.color_palette("tab20")[1],s=400)
    plt.scatter(Xdf[attr][bl_vec1], np.zeros(Xdf[attr][bl_vec1].shape),label='T%dP'%(k+1), marker='*',color='k',s=600, facecolors='none')
    plt.legend(fontsize=16, loc='lower right' )
    plt.title(lbx[k], fontsize=24)
    plt.xlabel(attr.replace('_','_'), fontsize=22)
    # plt.ylabel("Differences (kbps)", fontsize=14)
    # plt.xticks(np.arange(0,8,2), np.arange(0,8,2), fontsize=18)
    # plt.yticks(fontsize=18)
    plt.yticks([])
    # ·plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
        ax.spines[axis].set_color('k')  # change color
    k = k+1
    plt.savefig('data/figures/satimage_feat_anomalies_%dN.png'%(k),dpi=150, bbox_inches='tight')
    # plt.savefig('figs/problematic_features_outliers_%dFV.png'%(k),dpi=150, bbox_inches='tight')
    plt.show()


cross_val_param  = 6
n_leafs          = 1
depth            = 10
X = Xdf.copy()
X['Problem'] = problem
X.loc[X.Problem.str.contains('^[^P]+$'), "Problem"] = "Compliant"    
fX = X.copy()
fy = fX["Problem"]
fX = fX.drop(["Problem",], axis=1)


m3 = DTClassifier(depth, n_leafs, cross_val_param)
# m3 = DecisionTreeClassifier(max_depth=4)
m3.fit(fX, fy)
yo = m3.predict(fX)

dot_data = tree.export_graphviz(
#                                 m3,
                                m3.model,
                                feature_names = np.array(fX.columns), 
                                class_names=m3.model.classes_, 
#                                 class_names=m3.classes_, 
                                filled=True, 
                                rounded=True, 
                                out_file=None,
                                precision=6,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.format='png'
filename = graph.render(filename='models/telia/classification_model_test')







data = pd.read_csv('data/clean_data/' + data_file , sep=",", decimal='.', low_memory = False)
print('Data Size:(%d, %d)'%(data.shape[0], data.shape[1]))
print('-----------------------------------------')
target_kpi = 'transfer.datarate'
kpi_aux    = data[target_kpi]


data = data.loc[:,data.columns != target_kpi]
data = data.loc[:,data.columns != 'abs.uplinkdelay.avg']
data[target_kpi] = np.log10(kpi_aux)

scaler    = StandardScaler()
detector0 = OneClassSVM(gamma='auto', kernel='poly', degree=3).fit(scaler.fit_transform(data))
anomaly0  = detector0.predict(scaler.fit_transform(data))
if (np.sum(anomaly0 == 1) > np.sum(anomaly0 == -1)):
    anomaly0 = -anomaly0
outliers_features = (anomaly0 == 1)
print('Outlier Rate in Features: %.4f'%(np.mean(outliers_features)))


# outlier_matrix = np.zeros(data.shape)
# for ii, col in enumerate(data.columns):
#     if ((np.std(data[col]) != 0) & (len(data[col].unique()) > 0.1 *data.shape[0])):
#         data_nrm = data[col].to_numpy() / np.std(data[col])
#         Q3 = np.quantile(data_nrm, 0.75) - np.median(data_nrm)
#         if (Q3 > 0):
#             med = np.median(data[col])
#             MAD = (1/Q3) * np.median(np.absolute(data[col] - med))
#             outlier_matrix[:,ii] = (data[col] > med + 3.0*MAD)|(data[col] < med - 3.0*MAD)
# outliers_features = np.any(outlier_matrix, axis=1)
# print('Outlier Rate in Features: %.4f'%(np.mean(outliers_features)))


# detector1 = OneClassSVM(gamma='auto', kernel='poly', degree=3).fit(data[target_kpi].to_numpy().reshape(-1,1))
# anomaly1  = detector1.predict(data[target_kpi].to_numpy().reshape(-1,1))
# if (np.sum(anomaly1 == 1) > np.sum(anomaly1 == -1)):
#     anomaly1 = -anomaly1
# outliers_kpi = (anomaly1 == 1)
# print('Outlier Rate in KPI: %.4f'%(np.mean(outliers_kpi)))



data_nrm = data[target_kpi].to_numpy() / np.std(data[target_kpi])
Q3 = np.quantile(data_nrm, 0.75) - np.median(data_nrm)
if (Q3 != 0):
    med = np.median(data[target_kpi])
    MAD = (1/Q3) * np.median(np.absolute(data[target_kpi] - med))
    outliers_kpi = (data[target_kpi] > med + 3.0*MAD)|(data[target_kpi] < med - 3.0*MAD)
print('Outlier Rate in KPI: %.4f'%(np.mean(outliers_kpi)))


fig = plt.figure(dpi=300)
ax = fig.add_subplot()
ax.hist(kpi_aux.to_numpy()*0.000001, 
        bins=30, 
        density = False,
        color=sns.color_palette("tab20")[1],
        edgecolor='w')
plt.tick_params(axis='both', labelsize=22)
# plt.legend(fontsize=22)
plt.xlabel('transfer.datarate x 1e6', fontsize=26)
plt.ylabel('Count', fontsize=26)
plt.title('(a)',fontsize=26)
plt.autoscale(axis='both', tight=True)
plt.savefig('data/figures/OD_hist_KPI_example.png', bbox_inches='tight' )
plt.show()


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
        edgecolor='w')
plt.tick_params(axis='both', labelsize=22)
plt.legend(fontsize=22)
plt.xlabel('log10(transfer.datarate)', fontsize=26)
plt.ylabel('Count', fontsize=26)
plt.title('(b)',fontsize=26)
plt.autoscale(axis='both', tight=True)
plt.savefig('data/figures/OD_hist_logKPI_example.png', bbox_inches='tight' )
plt.show()


data_normality  = data.copy()
data_normality  = data[~(outliers_features|outliers_kpi)]
data_normality[target_kpi] = kpi_aux[~(outliers_features|outliers_kpi)].to_numpy().reshape(-1, 1)

train_cols = data_normality.columns[data_normality.columns != target_kpi]
X_train    = data_normality.loc[:, train_cols]
y_train    = data_normality[target_kpi].to_numpy().reshape(-1, 1) 

if (regressor == 'XGB'):
    model_normality = xgb.XGBRegressor()
    model_normality.fit(X_train, y_train)
    y_predicted     = model_normality.predict(data[train_cols])
    
    explainer_normality_model = shap.TreeExplainer(model_normality)
    shap_values_regressor     = explainer_normality_model.shap_values(X_train)

    shap.summary_plot(shap_values_regressor, 
                      X_train, 
                      max_display=8, 
                      show=False, 
                      cmap='coolwarm',
                      color_bar=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.xaxis.get_label().set_fontsize(24)
    ax.yaxis.get_label().set_fontsize(24)
    ax.ticklabel_format(axis='x', style='sci')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.title('(a) ' + regressor + ' Normality Model', fontsize=28)
    
    m = cm.ScalarMappable(cmap='coolwarm')
    m.set_array([0, 1])
    cb =  plt.colorbar(m,ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature Value', size=20, labelpad=0)
    cb.ax.tick_params(labelsize=20, length=0)
    
    # cbar = fig.colorbar()
    # cbar.ax.tick_params(labelsize=16)
    plt.savefig('data/figures/shap_normal_xgb.png',dpi=150, bbox_inches='tight')
    plt.show()
if (regressor == 'LGB'):
    model_normality = lgb.LGBMRegressor()
    model_normality.fit(X_train, y_train)
    y_predicted     = model_normality.predict(data[train_cols])
    
    explainer_normality_model = shap.TreeExplainer(model_normality)
    shap_values_regressor     = explainer_normality_model.shap_values(X_train)

    shap.summary_plot(shap_values_regressor, 
                      X_train, 
                      max_display=8, 
                      show=False, 
                      cmap='coolwarm',
                      color_bar=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.xaxis.get_label().set_fontsize(24)
    ax.yaxis.get_label().set_fontsize(24)
    ax.ticklabel_format(axis='x', style='sci')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.title('(b) ' + regressor + 'M Normality Model', fontsize=28)
    
    m = cm.ScalarMappable(cmap='coolwarm')
    m.set_array([0, 1])
    cb =  plt.colorbar(m,ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature Value', size=20, labelpad=0)
    cb.ax.tick_params(labelsize=20, length=0)
    
    # cbar = fig.colorbar()
    # cbar.ax.tick_params(labelsize=16)
    plt.savefig('data/figures/shap_normal_lgb.png',dpi=150, bbox_inches='tight')
    plt.show()
    
    
y_predicted[y_predicted<=0] = 1e-3
y_difference_log10 = (np.log10(y_predicted) - np.log10(kpi_aux)).to_numpy().reshape(-1,1)
y_difference       = (y_predicted - kpi_aux).to_numpy().reshape(-1,1)
anomaly_detector   = OneClassSVM(gamma='auto', kernel='poly', degree=3).fit(y_difference_log10)
# anomalies = anomaly_detector.predict(y_difference_log10[~outliers_kpi])
anomalies_mad = anomaly_detector.predict(y_difference_log10)
if (np.sum(anomalies_mad == 1) > np.sum(anomalies_mad == -1)):
    anomalies_mad = -anomalies_mad
anomalies_mad = (anomalies_mad==1)
anomalies_original = anomalies_mad




# data_nrm = y_difference_log10[~outliers_kpi] / np.std(y_difference_log10[~outliers_kpi])
# Q3 = np.quantile(data_nrm, 0.75) - np.median(data_nrm)
# if (Q3 != 0):
#     med = np.median(y_difference_log10[~outliers_kpi])
#     MAD = (1/Q3) * np.median(np.absolute(y_difference_log10[~outliers_kpi] - med))
#     anomalies = (y_difference_log10[~outliers_kpi] > med + 3.0*MAD)|(y_difference_log10[~outliers_kpi] < med - 3.0*MAD)
# anomalies = anomalies.reshape(-1)


data_nrm = y_difference_log10 / np.std(y_difference_log10)
Q3 = np.quantile(data_nrm, 0.75) - np.median(data_nrm)
if (Q3 != 0):
    med = np.median(y_difference_log10)
    MAD = (1/Q3) * np.median(np.absolute(y_difference_log10 - med))
    anomalies = (y_difference_log10 > med + 3.0*MAD)|(y_difference_log10 < med - 3.0*MAD)
anomalies = anomalies.reshape(-1)
print('Anomaly Rate: %.4f'%(np.mean(anomalies)))

# kpi_aux_anm = kpi_aux[~outliers_kpi]
# y_difference_anm = y_difference[~outliers_kpi]

kpi_aux_anm = kpi_aux
y_difference_anm = y_difference

anomaly_rate = np.mean(anomalies)
random_state = 42


unsupervised_classifiers ={'IForest': IForest(contamination = anomaly_rate,
                                          bootstrap = True,
                                          random_state = random_state),
              'KNN':      KNN(contamination=anomaly_rate),
	          'LOF':      LOF(n_neighbors=35, 
                              contamination=anomaly_rate),
              'PCA':      PCA(contamination=anomaly_rate, 
                              random_state=random_state),
	          'GMM':      GMM(contamination=anomaly_rate,
							  random_state=random_state),
	          'KDE':      KDE(contamination=anomaly_rate),
              'CBLOF':    CBLOF(contamination=anomaly_rate,
			                    check_estimator=False, 
                                random_state=random_state),
              'COF':      COF(contamination=anomaly_rate,
                              n_neighbors=35,),
              'HBOS':     HBOS(n_bins='auto',
                               contamination=anomaly_rate),
              'COPOD':    COPOD(contamination=anomaly_rate),
              'ECOD':     ECOD(contamination=anomaly_rate),
              'LODA':     LODA(contamination=anomaly_rate),
}



ncols = 7
nrows = len(unsupervised_classifiers) + 1
fig = plt.figure(figsize=(3.50*ncols,3.50*nrows), tight_layout=True)
ax = fig.add_subplot(nrows,ncols,1)
plt.scatter(kpi_aux_anm[anomalies==False], y_difference_anm[anomalies==False], color=sns.color_palette("tab20")[1], s=20, label='Normality')
plt.scatter(kpi_aux_anm[anomalies==True], y_difference_anm[anomalies==True], color='k', marker='*', facecolors='none', s=80, label='Anomalies')
plt.xlabel('Real KPI', fontsize=18)
plt.ylabel('Differences', fontsize=18)
plt.title('(a) MAD', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


auc = roc_auc_score(anomalies, anomalies_mad)
f1scr = f1_score(anomalies, anomalies_mad)
ax = fig.add_subplot(nrows,ncols,2)
plt.scatter(kpi_aux_anm[anomalies_mad==False], y_difference_anm[anomalies_mad==False], color=sns.color_palette("tab20")[1], s=20, label='Normality')
plt.scatter(kpi_aux_anm[anomalies_mad==True], y_difference_anm[anomalies_mad==True], color='k', marker='*', facecolors='none', s=80, label='Anomalies')
plt.xlabel('Real KPI', fontsize=18)
plt.ylabel('Differences', fontsize=18)
plt.title('(b) OCSVM', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.text(0.15e6, -4e5, 'F1:%.2f'%(f1scr), fontsize=18)
plt.text(0.15e6, -5.5e5, 'AUC:%.2f'%(auc), fontsize=18)

lbls =['(c)','(d)','(e)','(f)',
       '(g)','(h)','(i)','(j)',
       '(k)','(l)','(m)','(n)',
       '(o)','(p)','(q)','(r)',]

X = y_difference_log10
for i, (clf_name, clf) in enumerate(unsupervised_classifiers.items()):
    print(i + 1, 'fitting', clf_name)
    clf.fit(X)
    anomalies_clf = clf.predict(X)
    
    auc = roc_auc_score(anomalies, anomalies_clf)
    f1scr = f1_score(anomalies, anomalies_clf)
    ax = fig.add_subplot(nrows,ncols,i + 1 + 2)
    plt.scatter(kpi_aux_anm[anomalies_clf==False], y_difference_anm[anomalies_clf==False], color=sns.color_palette("tab20")[1], s=20, label='Normality')
    plt.scatter(kpi_aux_anm[anomalies_clf==True], y_difference_anm[anomalies_clf==True], color='k', marker='*', facecolors='none', s=80, label='Anomalies')
    plt.xlabel('Real KPI (kbps)', fontsize=18)
    plt.ylabel('Differences (kbps)', fontsize=18)
    plt.title(lbls[i] + ' ' + clf_name, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.text(0.15e6, -4e5, 'F1:%.2f'%(f1scr), fontsize=18)
    plt.text(0.15e6, -5.5e5, 'AUC:%.2f'%(auc), fontsize=18)

fig.tight_layout()
plt.savefig('data/figures/anomaly_scatter.png',dpi=150, bbox_inches='tight')
plt.show()

# data1 = data.copy()
# data1 = data1.loc[(~outliers_features)|outliers_kpi, data1.columns != target_kpi]
# outliers_kpi_reduced_set = outliers_kpi[(~outliers_features)|outliers_kpi]
# ydiff1 = y_difference[(~outliers_features)|outliers_kpi]


data1 = data.copy()
data1 = data1.loc[:, data1.columns != target_kpi]
outliers_kpi_reduced_set = outliers_kpi
ydiff1 = y_difference


n_clusters   = int(np.log2(data1.shape[0])/2)

print('-----------------------------------------------------------------')
print('CLUSTER DENSITIES')
models, cls_mat = cluster_anomaly_support_optimal(data1, outliers_kpi_reduced_set, data1.columns, [], 'outlier1')
print(models)


attributes = models["Attribute"].values
attributes, att_ind = np.unique(attributes, return_index=True)
cls_mat = cls_mat[:,att_ind]
problem = np.empty(cls_mat.shape[0], dtype=object)
separator = ';'
for ii in range(cls_mat.shape[0]): 
    strings = []
    for jj in range(cls_mat.shape[1]):
        if ((cls_mat[ii,jj] == 1)&(outliers_kpi_reduced_set.to_numpy()[ii] == True)):
#         if (cls_mat[ii,jj] == 1):
            strings = strings + ['T%dP'%(jj+1)]
        if (cls_mat[ii,jj] == 2)|((cls_mat[ii,jj] == 1)&(outliers_kpi_reduced_set.to_numpy()[ii] == False)):
#         if (cls_mat[ii,jj] == 2):
            strings = strings + ['T%dN'%(jj+1)]
    problem[ii] = separator.join(strings)

lbx = ['(f)', '(d)', '(c)', '(e)']    
k = 0;
for attr in attributes:
    cls_vec = cls_mat[:,k]
    bl_vec1 = ((cls_vec==1)&(outliers_kpi_reduced_set.to_numpy()==True))
    bl_vec2 = (cls_vec==2)|((cls_vec==1)&(outliers_kpi_reduced_set.to_numpy()==False))
#     bl_vec1 = (cls_vec==1)
#     bl_vec2 = (cls_vec==2)
    fig = plt.figure(dpi=150,figsize=(4,3)) #Paper figure
    ax = fig.add_subplot()
    plt.scatter(data1[attr][bl_vec2], ydiff1[bl_vec2], label='T%dN'%(k+1), marker='o',color=sns.color_palette("tab20")[1],s=200)
    plt.scatter(data1[attr][bl_vec1], ydiff1[bl_vec1], label='T%dP'%(k+1), marker='*',color='k',s=300, facecolors='none')
    plt.legend(fontsize=16, loc='lower right' )
    plt.title(lbx[k], fontsize=24)
    plt.xlabel(attr.replace('_','\_'), fontsize=20)
    plt.ylabel("Differences (kbps)", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
        ax.spines[axis].set_color('k')  # change color
    k = k+1
    plt.savefig('data/figures/outliers1_%dN.png'%(k),dpi=150, bbox_inches='tight')
    # plt.savefig('figs/problematic_features_outliers_%dFV.png'%(k),dpi=150, bbox_inches='tight')
    plt.show()
    
cross_val_param  = 6
n_leafs          = 1
depth            = 10
X = data1.copy()
# X = X[attributes]
# X['y_diff'] = ydiff_outliers
X = X.loc[:,X.columns !=target_kpi]
# if (flag1):
#     X = X.loc[:,X.columns !=trg_col1]
X['Problem'] = problem
X.loc[X.Problem.str.contains('^[^P]+$'), "Problem"] = "Compliant"    
fX = X.copy()
fy = fX["Problem"]
fX = fX.drop(["Problem",], axis=1)

m3 = DTClassifier(depth, n_leafs, cross_val_param)
# m3 = DecisionTreeClassifier(max_depth=4)
m3.fit(fX, fy)
yo = m3.predict(fX)

dot_data = tree.export_graphviz(
#                                 m3,
                                m3.model,
                                feature_names = np.array(fX.columns), 
                                class_names=m3.model.classes_, 
#                                 class_names=m3.classes_, 
                                filled=True, 
                                rounded=True, 
                                out_file=None,
                                precision=6,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.format='png'
filename = graph.render(filename='models/telia/classification_model')







# data2 = data.copy()
# data2 = data2.loc[(~outliers_features)|anomalies, data2.columns != target_kpi]
# outliers_anm_reduced_set = anomalies[(~outliers_features)|anomalies]
# ydiff2 = y_difference[(~outliers_features)|anomalies]


data2 = data.copy()
data2 = data2.loc[:, data2.columns != target_kpi]
outliers_anm_reduced_set = anomalies
ydiff2 = y_difference


n_clusters   = int(np.log2(data1.shape[0])/2)

print('-----------------------------------------------------------------')
print('CLUSTER DENSITIES')
models, cls_mat = cluster_anomaly_support_optimal(data2, outliers_anm_reduced_set, data2.columns, [], 'anomalies1')
print(models)


attributes = models["Attribute"].values
attributes, att_ind = np.unique(attributes, return_index=True)
cls_mat = cls_mat[:,att_ind]
problem = np.empty(cls_mat.shape[0], dtype=object)
separator = ';'
for ii in range(cls_mat.shape[0]): 
    strings = []
    for jj in range(cls_mat.shape[1]):
        if ((cls_mat[ii,jj] == 1)&(outliers_anm_reduced_set[ii] == True)):
#         if (cls_mat[ii,jj] == 1):
            strings = strings + ['T%dP'%(jj+1)]
        if (cls_mat[ii,jj] == 2)|((cls_mat[ii,jj] == 1)&(outliers_anm_reduced_set[ii] == False)):
#         if (cls_mat[ii,jj] == 2):
            strings = strings + ['T%dN'%(jj+1)]
    problem[ii] = separator.join(strings)
    
k = 0;
lbx = ['(i)', '(g)', '(h)', '(j)']  
for attr in attributes:
    cls_vec = cls_mat[:,k]
    bl_vec1 = ((cls_vec==1)&(outliers_anm_reduced_set==True))
    bl_vec2 = (cls_vec==2)|((cls_vec==1)&(outliers_anm_reduced_set==False))
#     bl_vec1 = (cls_vec==1)
#     bl_vec2 = (cls_vec==2)
    fig = plt.figure(dpi=150,figsize=(4,3)) #Paper figure
    ax = fig.add_subplot()
    plt.scatter(data2[attr][bl_vec2], ydiff2[bl_vec2], label='T%dN'%(k+1), marker='o',color=sns.color_palette("tab20")[1],s=200)
    plt.scatter(data2[attr][bl_vec1], ydiff2[bl_vec1], label='T%dP'%(k+1), marker='*',color='k',s=300, facecolors='none')
    plt.legend(fontsize=16, loc='lower right' )
    plt.title(lbx[k], fontsize=24)
    plt.xlabel(attr.replace('_','\_'), fontsize=20)
    plt.ylabel("Differences (kbps)", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
        ax.spines[axis].set_color('k')  # change color
    k = k+1
    plt.savefig('data/figures/anomalies1_%dN.png'%(k),dpi=150, bbox_inches='tight')
    # plt.savefig('figs/problematic_features_outliers_%dFV.png'%(k),dpi=150, bbox_inches='tight')
    plt.show()
    
cross_val_param  = 6
n_leafs          = 1
depth            = 10
X = data2.copy()
# X = X[attributes]
# X['y_diff'] = ydiff_outliers
X = X.loc[:,X.columns !=target_kpi]
# if (flag1):
#     X = X.loc[:,X.columns !=trg_col1]
X['Problem'] = problem
X.loc[X.Problem.str.contains('^[^P]+$'), "Problem"] = "Compliant"    
fX = X.copy()
fy = fX["Problem"]
fX = fX.drop(["Problem",], axis=1)

m3 = DTClassifier(depth, n_leafs, cross_val_param)
# m3 = DecisionTreeClassifier(max_depth=4)
m3.fit(fX, fy)
yo = m3.predict(fX)

dot_data = tree.export_graphviz(
#                                 m3,
                                m3.model,
                                feature_names = np.array(fX.columns), 
                                class_names=m3.model.classes_, 
#                                 class_names=m3.classes_, 
                                filled=True, 
                                rounded=True, 
                                out_file=None,
                                precision=6,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.format='png'
filename = graph.render(filename='models/telia/classification_model2')