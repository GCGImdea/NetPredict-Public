from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import time


# # Ploting libraries
# import os
# # make Qt run in offscreen mode (must be set before importing cv2)
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# # optional: make matplotlib use non-GUI backend
# import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import colors

# Import PyOD models
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

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import shap


from sklearn.metrics import jaccard_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import OneClassSVM
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

import graphviz

# ---- config ----
DATA_PATH   = Path("output_datasets/data_clean3.csv")
OUT_DIR     = Path("output_datasets")
TARGET      = "dl.throughput.value"
META_COLS   = [
    "test", "limitation", "throughputlim", "latencylim", "packetlosslim",
    "dl.throughput.unit", "dl.latency.unit", "dl.retransmission.unit",
]
THROUGHPUT_LIMS = ["1000kbit", "2000kbit", "5000kbit", "10000kbit", "20000kbit", "50000kbit"]

MISSING_MAX  = 0.01
VIF_MAX      = 20.0
RANDOM_STATE = 42
REGRESSOR    = 'XGB'

def mad_outliers(X: pd.Series, threshold: float = 3.0):
    Xn = X/X.std(ddof=0)
    Q3 = Xn.quantile(0.75) - Xn.mean()
    median = X.median()
    mad = (1/Q3) * (X - median).abs().median()
    if mad == 0:
        return pd.Series([True] * len(X), index=X.index)
    else:
        return (X - median).abs() >= threshold * mad
    
def make_detectors(contamination: float) -> dict[str, object]:
    """Factory: returns a compact, readable detector dict for a given contamination."""
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
    """Common binary classification metrics in one place."""
    return {
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "f1-score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "jaccard": jaccard_score(y_true, y_pred),
    }


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
        lowest_bic = np.inf
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
        print('GMM Clustering')
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
        # plt.savefig('data/figures/' + file_save + '.png',dpi=150, bbox_inches='tight')
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





data_prepared   = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)

for lim in THROUGHPUT_LIMS:
    lim_indices = (data_prepared['limitation'].isna()| (data_prepared['throughputlim'] == lim))
    data_lim = data_prepared.loc[lim_indices]
    outliers_real = (data_lim['throughputlim'] == lim)
    outliers_mad = mad_outliers(np.log10(data_lim[TARGET]))
    contamination = (outliers_real).mean()

    X_lim = data_lim.select_dtypes("number")
    X_lim[TARGET] = np.log10(X_lim[TARGET])
    rows = []
    for name, clf in make_detectors(contamination).items():
        print(f"fitting {lim}: {name}")
        clf.fit(X_lim)
        y_pred = pd.Series(clf.predict(X_lim), index=X_lim.index)
        rows.append({"method": name, **score_table(outliers_real, y_pred)})

    rows.append({"method": "MAD", **score_table(outliers_real, outliers_mad)})
    comparison = pd.DataFrame(rows).sort_values("f1-score", ascending=False)
    print(comparison.to_string(index=False))
    print("-" * 65)
        
    feats = X_lim.columns.drop(TARGET)
    X_train, y_train = X_lim.loc[~outliers_mad, feats], X_lim.loc[~outliers_mad, TARGET]
    model_normality = xgb.XGBRegressor()
    model_normality.fit(X_train, y_train)
    y_predicted     = model_normality.predict(X_lim.loc[:, X_lim.columns != TARGET])

    y_predicted[y_predicted<=0] = 1e-3
    y_difference = (y_predicted - X_lim[TARGET]).to_numpy().reshape(-1,1)
    
    anomaly_detector = OneClassSVM(gamma='auto', kernel='poly', degree=3).fit(y_difference)
    anomalies = anomaly_detector.predict(y_difference)
    if (np.sum(anomalies == 1) > np.sum(anomalies == -1)):
        anomalies = -anomalies
    anomalies = (anomalies==1)


    explainer_normality_model = shap.TreeExplainer(model_normality)
    shap_values_regressor     = explainer_normality_model.shap_values(X_train)

    shap.summary_plot(shap_values_regressor, 
                    X_train, 
                    max_display=12, 
                    show=False, 
                    cmap='coolwarm',
                    color_bar=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.xaxis.get_label().set_fontsize(24)
    ax.yaxis.get_label().set_fontsize(24)
    ax.ticklabel_format(axis='x', style='sci')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.title(REGRESSOR + ' Regressor. Limitation: ' + lim, fontsize=24)        
    m = cm.ScalarMappable(cmap='coolwarm')
    m.set_array([0, 1])
    cb =  plt.colorbar(m,ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature Value', size=20, labelpad=0)
    cb.ax.tick_params(labelsize=20, length=0)
    plt.show()
    
    xgb.plot_importance(model_normality,max_num_features=12)
    plt.title('XGB Feature Importance. Lim: ' + lim)
    plt.show()
    
    data1 = X_lim.copy()
    data1 = data1.loc[:,data1.columns != TARGET]
    
    models, cls_mat = cluster_anomaly_support_optimal(data1, anomalies, data1.columns, [], 'outlier1')

    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    ax[0].scatter(X_lim.loc[outliers_mad, TARGET], y_predicted[outliers_mad], alpha=0.5, color="red", label="Test")
    ax[0].scatter(X_lim.loc[~outliers_mad, TARGET], y_predicted[~outliers_mad], alpha=0.5, color="blue", label="Training")
    ax[0].set(xlabel=TARGET, ylabel="Predicted", title=f"Predicted vs Actual. Lim: {lim}")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].scatter(X_lim.loc[outliers_mad, TARGET], y_difference[outliers_mad], alpha=0.5, color="red", label="Test")
    ax[1].scatter(X_lim.loc[~outliers_mad, TARGET], y_difference[~outliers_mad], alpha=0.5, color="blue", label="Training")
    ax[1].set(xlabel=TARGET, ylabel="Differences", title=f"Differences vs Actual. Lim: {lim}")
    ax[1].grid(True)
    ax[1].legend()
    
    ax[2].scatter(X_lim.loc[anomalies, TARGET], y_difference[anomalies], alpha=0.5, color="red", label="Anomalies")
    ax[2].scatter(X_lim.loc[~anomalies, TARGET], y_difference[~anomalies], alpha=0.5, color="blue", label="Normal")
    ax[2].set(xlabel=TARGET, ylabel="Differences", title=f"Anomalies. Lim: {lim}")
    ax[2].grid(True)
    ax[2].legend()
    
    fig.tight_layout()
    plt.show()
    
    
    print(models)


    attributes = models["Attribute"].values
    attributes, att_ind = np.unique(attributes, return_index=True)
    cls_mat = cls_mat[:,att_ind]
    problem = np.empty(cls_mat.shape[0], dtype=object)
    separator = ';'
    for ii in range(cls_mat.shape[0]): 
        strings = []
        for jj in range(cls_mat.shape[1]):
            if ((cls_mat[ii,jj] == 1)&(anomalies[ii] == True)):
    #         if (cls_mat[ii,jj] == 1):
                strings = strings + ['T%dP'%(jj+1)]
            if (cls_mat[ii,jj] == 2)|((cls_mat[ii,jj] == 1)&(anomalies[ii] == False)):
    #         if (cls_mat[ii,jj] == 2):
                strings = strings + ['T%dN'%(jj+1)]
        problem[ii] = separator.join(strings)
        
        
        
    cross_val_param  = 6
    n_leafs          = 1
    depth            = 10
    X = data1.copy()
    X = X.loc[:,X.columns !=TARGET]
    X['Problem'] = problem
    X.loc[X.Problem.str.contains('^[^P]+$'), "Problem"] = "Compliant"    
    fX = X.copy()
    fy = fX["Problem"]
    fX = fX.drop(["Problem",], axis=1)
    
    m3 = DTClassifier(depth, n_leafs, cross_val_param)
    # m3 = DecisionTreeClassifier(max_depth=4)
    m3.fit(fX, fy)
    
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
    filename = graph.render(filename=f'decision_trees/lim_{lim}')   
    

