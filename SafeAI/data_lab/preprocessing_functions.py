import numpy as np
import pandas as pd
from sklearn import mixture

META_COLS   = ["limitation", "throughputlim", "latencylim", "packetlosslim"]

def compute_vif_from_corr(X: pd.DataFrame) -> pd.Series:
    """Compute VIF using the inverse of the correlation matrix.

    This is typically faster than regressing each column against all others.

    Notes:
    - Requires no NaNs (impute first).
    - Uses a pseudo-inverse to tolerate singular correlation matrices.
    """

    if X.shape[1] == 0:
        return pd.Series(dtype=float)

    # Standardize (avoid scale issues, corr is scale-invariant but helps numeric stability)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0).replace(0.0, np.nan)
    Xs = Xs.fillna(0.0)

    corr = np.corrcoef(Xs.to_numpy(), rowvar=False)

    inv_corr = np.linalg.pinv(corr)  # robust vs singular matrices
    vifs = np.diag(inv_corr)

    return pd.Series(vifs, index=X.columns, name="vif")

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = df.columns.astype(str).str.lower()
    c = c.str.replace("_", ".", regex=False).str.replace(" ", ".", regex=False)
    df = df.copy()
    df.columns = c.str.replace(r"\.+", ".", regex=True)
    return df


def clean_numeric(X: pd.DataFrame, missing_max: float = 0.01) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < missing_max]
    X = X.fillna(X.median(numeric_only=True))
    return X.loc[:, X.std(ddof=0) != 0]  # drop constants


def keep_meta(out: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in META_COLS if c in meta.columns]
    return out.join(meta[cols]) if cols else out


def vif_prune(X: pd.DataFrame, vif_max: float = 20.0) -> pd.DataFrame:
    while X.shape[1] and (v := compute_vif_from_corr(X)).max() > vif_max:
        X = X.drop(columns=[v.idxmax()])
    return X

def mad_outliers(X: pd.Series, threshold: float = 3.0):
    Xn = X/X.std(ddof=0)
    Q3 = Xn.quantile(0.75) - Xn.mean()
    median = X.median()
    mad = (1/Q3) * (X - median).abs().median()
    if mad == 0:
        return pd.Series([True] * len(X), index=X.index)
    else:
        return (X - median).abs() >= threshold * mad
    
def bic_grid(x: np.ndarray, max_k: int, iterations: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo BIC grid:
    - x: (n, 1)
    - returns (bic_ite, seeds) where bic_ite shape is (iterations, max_k-1)
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**16, size=(iterations, max_k - 1))
    bic_ite = np.empty((iterations, max_k - 1), dtype=float)

    for i in range(iterations):
        for k in range(1, max_k):
            gmm = mixture.GaussianMixture(n_components=k, random_state=int(seeds[i, k - 1]))
            gmm.fit(x)
            bic_ite[i, k - 1] = gmm.bic(x)
    return bic_ite, seeds

def best_gmm_by_bic(x: np.ndarray, k: int, seeds: np.ndarray) -> mixture.GaussianMixture:
    """Fit k-component GMM over multiple seeds and return the lowest-BIC model."""
    best, best_bic = None, np.inf
    for s in seeds:
        gmm = mixture.GaussianMixture(n_components=k, random_state=int(s))
        gmm.fit(x)
        b = gmm.bic(x)
        if b < best_bic:
            best, best_bic = gmm, b
    assert best is not None
    return best