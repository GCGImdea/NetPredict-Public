from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing_functions import norm_cols, clean_numeric, keep_meta
from preprocessing_functions import vif_prune
from preprocessing_functions import best_gmm_by_bic, bic_grid

from plot_functions import plot_bic_and_clusters, plot_feature_scatter

# ---- config ----
DATA_PATH   = Path("data/aggregated_stats.csv")
OUT_DIR     = Path("output_datasets")
TARGET      = "dl.throughput.value"
META_COLS   = ["limitation", "throughputlim", "latencylim", "packetlosslim"]
THROUGHPUT_LIMS = ["1000kbit", "2000kbit", "5000kbit", "10000kbit", "20000kbit", "50000kbit"]

MISSING_MAX  = 0.01
VIF_MAX      = 20.0
RANDOM_STATE = 42

EVAL_WINDOW = 40
BIC_ITERATIONS  = 50

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = norm_cols(pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False))
    y = df[TARGET]
    X = clean_numeric(df.select_dtypes("number"), missing_max=MISSING_MAX)
    meta = df.select_dtypes("object")
    
    out1 =keep_meta(X, meta)
    out1.to_csv(OUT_DIR / "data_clean1.csv", index=False)
    print(f"Saved data_clean1.csv with {X.shape[1]} features (after cleaning, before VIF pruning)")
    
    X2 = vif_prune(X, vif_max=VIF_MAX)
    out2 = keep_meta(pd.concat([X2, y.rename(TARGET)], axis=1), meta)
    out2.to_csv(OUT_DIR / "data_clean2.csv", index=False)
    print(f"Saved data_clean2.csv with {X2.shape[1]} features (after VIF pruning)")
    
    out_num = out2.select_dtypes("number")
    corr_matrix = 1 - out_num.corr(method='pearson').abs()
    corr_vector = corr_matrix[TARGET].to_numpy().reshape(-1,1)
    
    # Monte Carlo Approximation to estimate the average BIC curve
    max_num_clusters = np.min([EVAL_WINDOW + 1, len(corr_vector)])
    bic_ite, seeds = bic_grid(corr_vector, max_k=max_num_clusters, iterations=BIC_ITERATIONS, seed=RANDOM_STATE)
        
    # Determining the optimal number of clusters
    num_opt_clusters_bic = np.argmin(np.mean(bic_ite, axis=0)) + 1
    best = best_gmm_by_bic(corr_vector, k=num_opt_clusters_bic, seeds=seeds[:, num_opt_clusters_bic - 1])
    clusters_bic = best.predict(corr_vector)
    centroids_bic = best.means_
    
    out3 = out_num.copy()
    if (len(np.unique(clusters_bic)) > 1):
        out3 = out3.loc[:, out3.columns[(clusters_bic != np.argmin(centroids_bic))]]
    out3[TARGET] = y
    out3 = keep_meta(out3, meta)
    out3.to_csv(OUT_DIR / "data_clean3.csv", index=False)
    print(f"Saved data_clean3.csv with {out3.shape[1]} features (after high-correlation pruning)")
    
    # compact plots (optional)
    plot_bic_and_clusters(bic_ite, 
                          BIC_ITERATIONS, 
                          max_num_clusters, 
                          num_opt_clusters_bic, 
                          corr_vector, 
                          clusters_bic)
    columns_to_remove_bic = out_num.columns[(clusters_bic == np.argmin(centroids_bic))]
    plot_feature_scatter(out_num, TARGET, columns_to_remove_bic, ncols= 4)

if __name__ == "__main__":
    main()