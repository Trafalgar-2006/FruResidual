"""Enrichment testing: is a subset of neurons concentrated in specific
categories (e.g. superclass) more than chance would predict, given the
background population it was drawn from?

Used by the Phase-0 gate to replace an arbitrary "top-2 categories sum to
>=50%" heuristic with a real statistical test (Fisher's exact test per
category vs. the background, Benjamini-Hochberg corrected across
categories) — the same approach the plan specifies for Phase 3's proper
enrichment analysis (docs/project_plan.md section d, PHASE 3), just
applied at a lighter scope here.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def category_enrichment(
    subset: pd.Series,
    background: pd.Series,
    alpha: float = 0.05,
    min_odds_ratio: float = 2.0,
) -> pd.DataFrame:
    """Tests each category value in `subset` for over-representation
    relative to `background` (subset should be a sub-population of
    background, e.g. residual neurons vs. all dimorphic neurons).

    Returns one row per category with counts, odds ratio, raw p-value,
    BH-adjusted p-value, and whether it passes both the significance and
    effect-size bar.
    """
    categories = background.value_counts().index
    n_subset = len(subset)
    n_background = len(background)

    rows = []
    for cat in categories:
        a = int((subset == cat).sum())  # in subset, in category
        b = n_subset - a  # in subset, not in category
        c = int((background == cat).sum()) - a  # in background-not-subset, in category
        d = (n_background - n_subset) - c  # in background-not-subset, not in category
        # p-value from the exact (uncorrected) table; odds ratio uses a
        # Haldane-Anscombe +0.5 correction so a zero cell (e.g. a category
        # that's 100% residual, c=0) gives a large-but-finite, JSON-safe
        # estimate instead of infinity.
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        rows.append(
            {
                "category": cat,
                "n_in_subset": a,
                "n_in_background": a + c,
                "pct_of_subset": round(100 * a / n_subset, 1) if n_subset else 0.0,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value")
    if len(result):
        result["p_adj"] = multipletests(result["p_value"], alpha=alpha, method="fdr_bh")[1]
    else:
        result["p_adj"] = []
    result["significant"] = (result["p_adj"] < alpha) & (result["odds_ratio"] >= min_odds_ratio)
    return result.reset_index(drop=True)
