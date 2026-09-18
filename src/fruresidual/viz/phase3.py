"""Plotting for Phase 3: motif enrichment and connectivity clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

_SIG_COLOR = "#d1495b"
_NONSIG_COLOR = "#bfc3c7"


def plot_motif_enrichment(
    motif_result: pd.DataFrame, out_path: Path, top_n: int = 20, degree_confounded_strata: set[str] | None = None
) -> None:
    """`motif_result` must have a "superclass" column (the stratum each
    row's test ran within -- see scripts/run_phase3.py) so bars are
    labeled by stratum, not just partner type: a combined top-N-by-
    p-value chart with unlabeled bars would silently hide which stratum
    each result came from, an easy way to mislead a reader into treating
    every bar as equally trustworthy.

    Plots the DEGREE-ADJUSTED statistics when they are present (they are,
    since scripts/run_phase3.py started running the Cochran-Mantel-
    Haenszel test) and falls back to the marginal ones otherwise. This
    matters: plotting marginal odds ratios next to a degree-adjusted
    shortlist would show the reader a chart of numbers that no longer
    back the conclusions drawn from it.
    """
    degree_confounded_strata = degree_confounded_strata or set()
    adjusted = "p_adj_degree_adjusted" in motif_result.columns
    p_col = "p_adj_degree_adjusted" if adjusted else "p_adj"
    or_col = "odds_ratio_degree_adjusted" if adjusted else "odds_ratio"
    sig_col = "significant_degree_adjusted" if adjusted else "significant"

    df = motif_result.dropna(subset=[p_col]).sort_values(p_col).head(top_n).iloc[::-1].copy()
    df["label"] = df.apply(
        lambda r: f"{r['category']} [{r['superclass']}]"
        + (" †" if r["superclass"] in degree_confounded_strata else ""),
        axis=1,
    )
    colors = [_SIG_COLOR if s else _NONSIG_COLOR for s in df[sig_col].fillna(False)]

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.3 * len(df))))
    ax.barh(df["label"], df[or_col], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel(
        ("Degree-adjusted odds ratio (CMH, residual vs. explained within stratum)" if adjusted
         else "Odds ratio (residual vs. explained, within stratum)") + ", log scale"
    )
    ax.set_title(
        f"Connectivity-motif enrichment: top {top_n} by p-value\n"
        + ("adjusted for connectivity degree (CMH)\n† = degree differs between groups in this stratum"
           if adjusted else "† = stratum has a detected degree confound (see writeup)"),
        fontsize=10,
    )
    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_scatter(
    matrix: pd.DataFrame, cluster_labels: pd.Series, residual_body_ids: set[int], out_path: Path
) -> None:
    """2D PCA projection of the connectivity-profile matrix, colored by
    Leiden cluster, with residual neurons outlined -- lets you see at a
    glance whether the residual concentrates in specific clusters."""
    coords = PCA(n_components=2, random_state=0).fit_transform(matrix.values)
    is_residual = matrix.index.isin(residual_body_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cluster_labels.reindex(matrix.index).values,
        cmap="tab20", s=14, alpha=0.7,
    )
    ax.scatter(
        coords[is_residual, 0], coords[is_residual, 1],
        facecolors="none", edgecolors="black", s=40, linewidths=0.8, label="residual (fru/dsx-)",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Connectivity-profile clusters (color) with residual neurons outlined")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Leiden cluster")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
