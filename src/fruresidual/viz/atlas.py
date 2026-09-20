"""Plotting for the Phase-1 unexplained-dimorphism atlas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SIG_COLOR = "#d1495b"
_NONSIG_COLOR = "#bfc3c7"


def plot_enrichment_bar(enrichment_df: pd.DataFrame, title: str, out_path: Path, top_n: int = 15) -> None:
    """Bar chart of odds ratio per category, colored by statistical
    significance (the real test, not a raw-count heuristic).
    """
    df = enrichment_df.head(top_n).iloc[::-1]
    colors = [_SIG_COLOR if s else _NONSIG_COLOR for s in df["significant"]]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(df))))
    ax.barh(df["category"], df["odds_ratio"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (residual vs. background), log scale")
    ax.set_title(title)
    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roi_bar(roi_summary: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    """Bar chart of %-residual synapse weight per ROI, sorted by total
    involvement (so it's not dominated by tiny, noisy ROIs). This ranking
    answers "which regions carry the most residual in absolute terms" --
    see plot_roi_bar_by_pct for "which regions are most disproportionately
    unexplained," a different and equally important question.
    """
    df = roi_summary.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(df))))
    ax.barh(df.index, df["pct_residual"], color=_SIG_COLOR)
    ax.set_xlabel("% of dimorphic-neuron synapse weight that is fru/dsx-unexplained")
    ax.set_xlim(0, 100)
    # Wrapped and down-sized: at figsize=7 wide this title overflows the
    # axes and gets clipped mid-word, which is then baked into the
    # summary panel that embeds this figure.
    ax.set_title(
        f"Unexplained-dimorphism residual by neuropil\n(top {top_n} by total involvement)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roi_bar_by_pct(
    roi_summary: pd.DataFrame, out_path: Path, top_n: int = 20, min_total_synweight: int = 100
) -> None:
    """Bar chart of ROIs ranked by %-residual (not total involvement),
    filtered to a minimum synapse-weight floor so it isn't dominated by
    tiny, noisy ROIs with only a handful of synapses.
    """
    df = roi_summary[roi_summary["total_synweight"] >= min_total_synweight]
    df = df.sort_values("pct_residual", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(df))))
    ax.barh(df.index, df["pct_residual"], color=_SIG_COLOR)
    ax.set_xlabel("% of dimorphic-neuron synapse weight that is fru/dsx-unexplained")
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Most disproportionately unexplained neuropils (top {top_n}, "
        f"min {min_total_synweight} synapse weight)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
