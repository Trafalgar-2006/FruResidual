"""Plotting for Phase 4: cross-dataset concordance with BANC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_concordance_bar(overall: dict, by_superclass: dict, out_path: Path) -> None:
    """Bar chart of %-confirmed-dimorphic-by-BANC, residual vs. explained,
    overall and per superclass -- the plan's requested "reproducibility
    figure showing which residual findings survive a second female
    dataset." Superclasses with too few matches to test are shown with
    a hatch pattern so a thin bar isn't mistaken for a strong result.
    """
    rows = [("overall", overall)]
    for sc, res in by_superclass.items():
        if "pct_residual_confirmed_dimorphic_by_banc" in res:
            rows.append((sc, res))

    labels = [r[0] for r in rows]
    residual_pct = [r[1]["pct_residual_confirmed_dimorphic_by_banc"] or 0 for r in rows]
    explained_pct = [r[1]["pct_explained_confirmed_dimorphic_by_banc"] or 0 for r in rows]
    n_residual = [r[1]["n_residual_matched"] for r in rows]
    n_explained = [r[1]["n_explained_matched"] for r in rows]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], residual_pct, width, label="residual (fru/dsx-)", color="#d1495b")
    ax.bar([i + width / 2 for i in x], explained_pct, width, label="explained (fru/dsx+)", color="#6c8ebf")
    for i, (nr, ne) in enumerate(zip(n_residual, n_explained, strict=True)):
        ax.text(i - width / 2, residual_pct[i] + 2, f"n={nr}", ha="center", fontsize=8)
        ax.text(i + width / 2, explained_pct[i] + 2, f"n={ne}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% independently confirmed dimorphic by BANC")
    ax.set_ylim(0, 110)
    ax.set_title("Cross-dataset validation: does BANC confirm MaleCNS's dimorphism calls?\n"
                 "(per-stratum breakdowns are small-n and underpowered -- see writeup)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
