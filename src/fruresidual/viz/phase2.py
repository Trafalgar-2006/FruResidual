"""Plotting for Phase 2: the predictive model and ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_COLORS = {"without_fru_dsx": "#6c8ebf", "with_fru_dsx": "#d1495b"}


def plot_ablation_comparison(ablation_table: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: R^2 per model, with vs. without fru/dsx. The
    gap between bars for the same model IS the headline Phase-2 number
    (how much the genetic switch explains)."""
    models = ablation_table["model"].unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    for i, ablation in enumerate(["without_fru_dsx", "with_fru_dsx"]):
        sub = ablation_table[ablation_table["ablation"] == ablation].set_index("model").loc[models]
        x = [j + i * width for j in range(len(models))]
        ax.bar(x, sub["r2_mean"], width, yerr=sub["r2_std"], label=ablation, color=_COLORS[ablation])
    ax.set_xticks([j + width / 2 for j in range(len(models))])
    ax.set_xticklabels(models)
    ax.set_ylabel("Cross-validated R^2")
    ax.set_title("Ablation: how much does fru/dsx explain wiring difference?")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(residual_scores: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = residual_scores["is_case"].map({True: "#d1495b", False: "#6c8ebf"})
    ax.scatter(
        residual_scores["actual_wiring_difference"],
        residual_scores["predicted_wiring_difference"],
        c=colors, alpha=0.5, s=12,
    )
    lims = [0, 1]
    ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Actual wiring difference (cosine distance)")
    ax.set_ylabel("Predicted (out-of-fold)")
    ax.set_title("Predicted vs. actual — red=dimorphic case, blue=isomorphic control")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual_distribution(residual_scores: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        residual_scores.loc[~residual_scores["is_case"], "residual"],
        bins=30, alpha=0.6, label="isomorphic (control)", color="#6c8ebf",
    )
    ax.hist(
        residual_scores.loc[residual_scores["is_case"], "residual"],
        bins=30, alpha=0.6, label="dimorphic (case)", color="#d1495b",
    )
    ax.set_xlabel("Residual (actual - predicted wiring difference)")
    ax.set_ylabel("count")
    ax.set_title("Per-neuron residual score distribution")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
