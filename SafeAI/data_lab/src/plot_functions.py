from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import shap
import xgboost as xgb

from src.config import REGRESSOR


# ── Preprocessing plots ───────────────────────────────────────────────────────

def plot_bic_and_clusters(
    bic_ite: np.ndarray,
    iterations: int,
    max_num_clusters: int,
    k_opt: int,
    x: np.ndarray,
    clusters: np.ndarray,
) -> None:
    """Plot Monte Carlo BIC curves and the resulting 1-D GMM clustering.

    Left panel: BIC vs number of clusters for each iteration + average.
    Right panel: scatter of correlation values coloured by cluster assignment.
    """
    meanbic = np.mean(bic_ite, axis=0)
    minbic  = np.min(meanbic)
    maxbic  = np.max(meanbic)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for ii in range(iterations):
        label = "Realization" if ii == 0 else None
        ax[0].plot(np.arange(1, max_num_clusters), bic_ite[ii, :],
                   lw=2, c="lightblue", label=label)
    ax[0].plot(np.arange(1, max_num_clusters), meanbic,
               "-o", lw=2, c="darkblue", label="Average")
    ax[0].plot([k_opt, k_opt], [minbic, maxbic], "k--", lw=2)
    ax[0].set_xlabel("Number of clusters", fontsize=14)
    ax[0].set_ylabel("BIC", fontsize=14)
    ax[0].tick_params(axis="both", labelsize=12)
    ax[0].set_ylim([minbic - 20, maxbic])
    ax[0].set_xlim([0, 40])
    ax[0].legend(fontsize=12, loc="lower right")

    for i in range(len(np.unique(clusters))):
        mask = clusters == i
        ax[1].scatter(x[mask], np.zeros(mask.sum()), s=100)
    ax[1].set_xlabel("Correlation factors", fontsize=14)
    ax[1].tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    plt.show()


def plot_feature_scatter(
    out_num: pd.DataFrame,
    target: str,
    cols: pd.Index,
    ncols: int = 5,
) -> None:
    """Scatter plot of each feature in `cols` against `target`.

    Lays out subplots in a grid with `ncols` columns, filling rows as needed.
    Empty subplots at the end are hidden automatically.
    """
    n     = len(cols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 2.5 * nrows),
                             tight_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for ii, col in enumerate(cols):
        r, c = divmod(ii, ncols)
        axes[r, c].plot(out_num[col], out_num[target], "o", color="tab:red")
        axes[r, c].set_xlabel(col, fontsize=18)
        axes[r, c].set_ylabel("KPI (kbps)", fontsize=18)

    # hide unused subplots
    for jj in range(n, nrows * ncols):
        r, c = divmod(jj, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    plt.show()


# ── Anomaly detection plots ───────────────────────────────────────────────────

def plot_shap_summary(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    lim: str,
) -> None:
    """SHAP beeswarm summary plot for an XGBoost regressor.

    Parameters
    ----------
    model   : fitted XGBRegressor
    X_train : training feature matrix used to compute SHAP values
    lim     : throughput limit label used in the plot title
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train,
                      max_display=12, show=False,
                      cmap="coolwarm", color_bar=False)

    fig, ax = plt.gcf(), plt.gca()
    ax.xaxis.get_label().set_fontsize(24)
    ax.yaxis.get_label().set_fontsize(24)
    ax.ticklabel_format(axis="x", style="sci")
    plt.tick_params(axis="x", labelsize=22)
    plt.tick_params(axis="y", labelsize=22)
    plt.title(f"{REGRESSOR} Regressor — Limitation: {lim}", fontsize=24)

    m = cm.ScalarMappable(cmap="coolwarm")
    m.set_array([0, 1])
    cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", size=20, labelpad=0)
    cb.ax.tick_params(labelsize=20, length=0)

    plt.show()


def plot_xgb_importance(model: xgb.XGBRegressor, lim: str) -> None:
    """Bar chart of XGBoost feature importances (top 12 features).

    Parameters
    ----------
    model : fitted XGBRegressor
    lim   : throughput limit label used in the plot title
    """
    xgb.plot_importance(model, max_num_features=12)
    plt.title(f"XGB feature importance — Lim: {lim}")
    plt.show()


def plot_road_ranking(
    all_ratios: np.ndarray,
    all_featrs: np.ndarray,
    num_features: int = 12,
) -> None:
    """Horizontal bar chart of ROAD feature ranking for anomalies.

    Parameters
    ----------
    all_ratios   : sorted Jaccard ratios (descending)
    all_featrs   : feature names aligned with all_ratios
    num_features : how many top features to display
    """
    if len(all_featrs) < num_features:
        return

    fig = plt.figure(figsize=(0.9 * 4, 0.9 * 3))
    plt.barh(
        np.arange(num_features, 0, -1),
        all_ratios[:num_features],
        height=0.50,
        color=sns.color_palette("tab20")[1],
        edgecolor=sns.color_palette("tab20")[0],
    )
    plt.yticks(np.arange(num_features, 0, -1),
               all_featrs[:num_features], fontsize=11)
    plt.title("ROAD ranking (anomalies)", fontsize=14)
    plt.show()


def plot_prediction_diagnostics(
    X_lim: pd.DataFrame,
    target: str,
    y_predicted: np.ndarray,
    y_difference: np.ndarray,
    outliers_mad: pd.Series,
    anomalies: pd.Series,
    lim: str,
) -> None:
    """Three-panel diagnostic plot for the normality regression model.

    Panels
    ------
    1. Predicted vs actual (MAD outliers highlighted in red)
    2. Residuals vs actual (MAD outliers highlighted in red)
    3. Anomalies vs actual (OneClassSVM anomalies highlighted in red)
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].scatter(X_lim.loc[outliers_mad, target],  y_predicted[outliers_mad],
                  alpha=0.5, color="red",  label="Test")
    ax[0].scatter(X_lim.loc[~outliers_mad, target], y_predicted[~outliers_mad],
                  alpha=0.5, color="blue", label="Training")
    ax[0].set(xlabel=target, ylabel="Predicted",
              title=f"Predicted vs actual — Lim: {lim}")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].scatter(X_lim.loc[outliers_mad, target],  y_difference[outliers_mad],
                  alpha=0.5, color="red",  label="Test")
    ax[1].scatter(X_lim.loc[~outliers_mad, target], y_difference[~outliers_mad],
                  alpha=0.5, color="blue", label="Training")
    ax[1].set(xlabel=target, ylabel="Residuals",
              title=f"Residuals vs actual — Lim: {lim}")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].scatter(X_lim.loc[anomalies, target],  y_difference[anomalies],
                  alpha=0.5, color="red",  label="Anomalies")
    ax[2].scatter(X_lim.loc[~anomalies, target], y_difference[~anomalies],
                  alpha=0.5, color="blue", label="Normal")
    ax[2].set(xlabel=target, ylabel="Residuals",
              title=f"Anomalies — Lim: {lim}")
    ax[2].grid(True)
    ax[2].legend()

    fig.tight_layout()
    plt.show()
    
# ── Safe AI plots ───────────────────────────────────────────────────
   
    
def plot_rga_by_limitation(
    rga_scores: dict[str, float],
) -> None:
    lims   = list(rga_scores.keys())
    values = list(rga_scores.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(lims, values,
                color=sns.color_palette("Blues_d", len(lims)),
                edgecolor="white", linewidth=0.8)

    ax.set_ylim([min(values) - 0.02, 1.01])
    ax.set_xlabel("Throughput limitation", fontsize=13)
    ax.set_ylabel("RGA", fontsize=13)
    ax.set_title("RGA — normality model accuracy per limitation", fontsize=14)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.tick_params(axis="both", labelsize=11)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.show()
    
def plot_rge_ranking(
    rge_df: pd.DataFrame,
    lim: str,
    top_n: int = 12,
) -> None:
    df = rge_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(6, 0.5 * top_n + 1))
    ax.barh(df["feature"][::-1], df["rge"][::-1],
            color=sns.color_palette("tab20")[3],
            edgecolor=sns.color_palette("tab20")[2],
            height=0.6)
    ax.set_xlabel("RGE", fontsize=13)
    ax.set_title(f"RGE ranking — Lim: {lim}", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    plt.show()
    
def plot_rgr_ranking(
    rgr_df: pd.DataFrame,
    lim: str,
    top_n: int = 12,
) -> None:
    df = rgr_df.tail(top_n).sort_values("rgr").copy()

    fig, ax = plt.subplots(figsize=(6, 0.5 * top_n + 1))
    ax.barh(df["feature"], df["rgr"],
            color=sns.color_palette("tab20")[5],
            edgecolor=sns.color_palette("tab20")[4],
            height=0.6)
    ax.set_xlabel("RGR", fontsize=13)
    ax.set_title(f"RGR — most sensitive features — Lim: {lim}", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    plt.show()
    
def plot_rge_vs_rgr(
    rge_df: pd.DataFrame,
    rgr_df: pd.DataFrame,
    lim: str,
    top_n: int = 15,
) -> None:
    merged = rge_df.merge(rgr_df, on="feature")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["rge"], merged["rgr"],
               alpha=0.5, color="steelblue", s=40, zorder=3)

    for _, row in merged.head(top_n).iterrows():
        ax.annotate(row["feature"],
                    xy=(row["rge"], row["rgr"]),
                    xytext=(6, 2), textcoords="offset points",
                    fontsize=8, color="dimgray")

    rge_mid = merged["rge"].median()
    rgr_mid = merged["rgr"].median()
    ax.axvline(rge_mid, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(rgr_mid, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("RGE — feature importance", fontsize=13)
    ax.set_ylabel("RGR — robustness", fontsize=13)
    ax.set_title(f"RGE vs RGR — Lim: {lim}", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    plt.show()
    
def plot_shap_vs_rge(
    shap_ranking: pd.DataFrame,
    rge_df: pd.DataFrame,
    lim: str,
    top_n: int = 15,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * top_n + 2), sharey=False)
    palette   = sns.color_palette("tab20")

    shap_plot = shap_ranking.head(top_n).copy()
    axes[0].barh(shap_plot["feature"][::-1], shap_plot["mean_abs_shap"][::-1],
                 color=palette[1], edgecolor=palette[0], height=0.6)
    axes[0].set_xlabel("Mean |SHAP|", fontsize=12)
    axes[0].set_title("SHAP ranking", fontsize=13)
    axes[0].tick_params(axis="both", labelsize=10)

    rge_plot = rge_df.head(top_n).copy()
    axes[1].barh(rge_plot["feature"][::-1], rge_plot["rge"][::-1],
                 color=palette[3], edgecolor=palette[2], height=0.6)
    axes[1].set_xlabel("RGE", fontsize=12)
    axes[1].set_title("RGE ranking", fontsize=13)
    axes[1].tick_params(axis="both", labelsize=10)

    fig.suptitle(f"SHAP vs RGE — Lim: {lim}", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()
    
def plot_rge_heatmap(
    results: dict,
    throughput_lims: list[str],
    top_n: int = 15,
) -> None:
    # raccoglie i top_n features per RGE dalla prima limitazione come riferimento
    ref_features = results[throughput_lims[0]]["safeai"]["rge"]["feature"].head(top_n).tolist()

    # costruisce una matrice features × limitazioni
    matrix = pd.DataFrame(index=ref_features, columns=throughput_lims, dtype=float)
    for lim in throughput_lims:
        rge_df = results[lim]["safeai"]["rge"].set_index("feature")
        for feat in ref_features:
            matrix.loc[feat, lim] = rge_df.loc[feat, "rge"] if feat in rge_df.index else 0.0

    fig, ax = plt.subplots(figsize=(len(throughput_lims) * 1.4 + 2, top_n * 0.5 + 1))
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".4f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "RGE", "shrink": 0.8},
    )
    ax.set_xlabel("Throughput limitation", fontsize=13)
    ax.set_ylabel("Feature", fontsize=13)
    ax.set_title("RGE per feature across limitations", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.show()