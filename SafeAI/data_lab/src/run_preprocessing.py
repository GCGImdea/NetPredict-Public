from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH, OUT_DATASETS,
    TARGET,
    MISSING_MAX, VIF_MAX,
    EVAL_WINDOW, BIC_ITERATIONS, RANDOM_STATE,
)
from src.preprocessing_functions import (
    norm_cols, clean_numeric, keep_meta,
    vif_prune, best_gmm_by_bic, bic_grid,
)


def main() -> None:
    OUT_DATASETS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & normalize ───────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = norm_cols(pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False))
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    y    = df[TARGET]
    X    = clean_numeric(df.select_dtypes("number"), missing_max=MISSING_MAX)
    meta = df.select_dtypes("object")

    # ── 2. Save data_clean1 (after basic cleaning) ────────────────────────────
    out1 = keep_meta(X, meta)
    out1.to_csv(OUT_DATASETS / "data_clean1.csv", index=False)
    print(f"Saved data_clean1.csv — {X.shape[1]} features")

    # ── 3. VIF pruning → data_clean2 ─────────────────────────────────────────
    X2   = vif_prune(X, vif_max=VIF_MAX)
    out2 = keep_meta(pd.concat([X2, y.rename(TARGET)], axis=1), meta)
    out2.to_csv(OUT_DATASETS / "data_clean2.csv", index=False)
    print(f"Saved data_clean2.csv — {X2.shape[1]} features (removed {X.shape[1] - X2.shape[1]})")

    # ── 4. Correlation-based pruning → data_clean3 ───────────────────────────
    out_num     = out2.select_dtypes("number")
    corr_matrix = 1 - out_num.corr(method="pearson").abs()
    corr_vector = corr_matrix[TARGET].to_numpy().reshape(-1, 1)

    max_num_clusters = min(EVAL_WINDOW + 1, len(corr_vector))
    bic_ite, seeds   = bic_grid(
        corr_vector,
        max_k=max_num_clusters,
        iterations=BIC_ITERATIONS,
        seed=RANDOM_STATE,
    )

    k_opt         = int(np.argmin(np.mean(bic_ite, axis=0))) + 1
    best          = best_gmm_by_bic(corr_vector, k=k_opt, seeds=seeds[:, k_opt - 1])
    clusters_bic  = best.predict(corr_vector)
    centroids_bic = best.means_

    print(f"Optimal clusters (BIC): {k_opt}")

    out3 = out_num.copy()
    if len(np.unique(clusters_bic)) > 1:
        keep_mask = clusters_bic != np.argmin(centroids_bic)
        removed   = (~keep_mask).sum()
        out3      = out3.loc[:, out3.columns[keep_mask]]
        print(f"Removed {removed} high-correlation features")

    out3[TARGET] = y
    out3         = keep_meta(out3, meta)
    out3.to_csv(OUT_DATASETS / "data_clean3.csv", index=False)
    print(f"Saved data_clean3.csv — {out3.shape[1]} total columns")


if __name__ == "__main__":
    main()
