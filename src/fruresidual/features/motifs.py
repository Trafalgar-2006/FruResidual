"""Phase 3: connectivity-motif enrichment -- does the residual set share
specific partner cell TYPES more than background, not just broad partner
regions (already covered at superclass granularity in Phase 1)? See
docs/project_plan.md section (d), PHASE 3, step 2.

At ~11,000+ distinct partner types across ~2,368 neurons, most types are
touched by only a handful of neurons -- testing all of them would mean
thousands of near-zero-power hypotheses diluting the BH correction to
uselessness. min_count filtering (a partner type must be touched by at
least min_count dimorphic neurons total) keeps the test set to types
that could plausibly reach significance, and is applied BEFORE testing
(not after, which would be circular/would inflate false positives).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def filter_by_min_count(
    partner_profile_long: pd.DataFrame, min_neurons: int = 10
) -> pd.DataFrame:
    """Keeps only partner types touched by at least `min_neurons` distinct
    neurons in the population (not min total weight -- a single neuron
    with one huge synapse shouldn't count as "shared")."""
    counts = partner_profile_long.groupby("partner_type")["bodyId"].nunique()
    keep = counts[counts >= min_neurons].index
    return partner_profile_long[partner_profile_long["partner_type"].isin(keep)]


def identify_mixed_strata(
    dimorphic: pd.DataFrame, residual_body_ids: set[int], stratum_col: str = "superclass", min_group_size: int = 20
) -> pd.DataFrame:
    """Identifies which values of `stratum_col` have enough of BOTH
    residual and explained neurons to support a within-stratum
    comparison (>= min_group_size of each).

    Necessary before any residual-vs-explained enrichment test: a
    stratum with zero internal residual/explained variance (e.g.
    ol_intrinsic was found to be 100% residual, 0% explained -- see
    test_regional_confound_produces_spurious_marginal_enrichment) makes
    any partner-type or cluster disproportionately touched by that
    stratum look spuriously "residual-enriched" in an unstratified
    (marginal) test -- tautologically, since being IN that stratum
    already implies being residual. Testing only within strata that have
    real variance avoids reporting that tautology as a finding.
    """
    d = dimorphic.copy()
    d["is_residual"] = d["bodyId"].isin(residual_body_ids)
    summary = d.groupby(stratum_col).agg(
        n_total=("bodyId", "size"), n_residual=("is_residual", "sum")
    )
    summary["n_explained"] = summary["n_total"] - summary["n_residual"]
    summary["pct_residual"] = (summary["n_residual"] / summary["n_total"] * 100).round(1)
    mixed = summary[(summary["n_residual"] >= min_group_size) & (summary["n_explained"] >= min_group_size)]
    return mixed.reset_index()


def check_degree_confound(
    partner_profile_long: pd.DataFrame, residual_body_ids: set[int], background_body_ids: set[int]
) -> dict:
    """Checks whether residual and explained neurons differ in overall
    connectivity intensity (distinct partner-type count, total weight) --
    a confound the per-category Fisher's tests in partner_type_enrichment
    do NOT control for. Found real on cb_intrinsic (2026-09-18): residual
    neurons had ~24% higher mean partner-type degree (Mann-Whitney
    p=0.0075) and ~1.7x higher total connectivity weight. A degree
    difference like this can inflate apparent enrichment broadly across
    many categories simply because higher-degree neurons are more likely
    to touch any given partner type -- so any significant motif result
    should be read as "candidate, not yet degree-adjusted" rather than
    proof of a specific circuit signature. Degree-matched case-control
    sampling or including degree as a covariate (e.g. logistic
    regression instead of marginal Fisher tests) would be the next step
    to fully control this, left as future work.
    """
    from scipy.stats import mannwhitneyu

    degree = partner_profile_long.groupby("bodyId")["partner_type"].nunique()
    weight = partner_profile_long.groupby("bodyId")["weight"].sum()
    explained_ids = background_body_ids - residual_body_ids

    residual_degree = degree.reindex(list(residual_body_ids)).dropna()
    explained_degree = degree.reindex(list(explained_ids)).dropna()
    _, p_value = mannwhitneyu(residual_degree, explained_degree)

    return {
        "residual_mean_degree": float(residual_degree.mean()),
        "explained_mean_degree": float(explained_degree.mean()),
        "residual_mean_weight": float(weight.reindex(list(residual_body_ids)).dropna().mean()),
        "explained_mean_weight": float(weight.reindex(list(explained_ids)).dropna().mean()),
        "degree_difference_p_value": float(p_value),
        "degree_confounded": bool(p_value < 0.05),
    }


def partner_type_enrichment(
    partner_profile_long: pd.DataFrame,
    residual_body_ids: set[int],
    background_body_ids: set[int],
    min_neurons: int = 10,
    alpha: float = 0.05,
    min_odds_ratio: float = 2.0,
) -> pd.DataFrame:
    """For each partner type (each connected neuron counts once, not
    weighted by synapse strength -- presence/absence of a connection is
    the "motif" being tested, matching the plan's "shared input/output
    partners" framing), tests whether residual neurons connect to it
    more than background (residual + explained) neurons do.

    Deliberately NOT built on fruresidual.eval.enrichment.category_enrichment:
    that function assumes single-label membership (one category value per
    row/entity, e.g. one superclass per neuron), but partner_type is
    multi-label -- one neuron connects to many partner types
    simultaneously, contributing many rows. Reusing category_enrichment's
    len(series) as "population size" silently counts EDGES, not NEURONS,
    which corrupts every odds ratio and p-value (verified 2026-09-18: a
    category touched by 29/215 = 13.5% of residual neurons was reported
    as touching 0.04% of "subset" because the denominator was ~80,000
    residual-neuron EDGES, not the 215 residual neurons). This builds
    the 2x2 table directly from neuron-level set membership per category
    instead, matching what "13.5% of residual neurons connect to X" is
    actually supposed to mean.

    residual_body_ids must be a subset of background_body_ids (background
    = the whole comparison population, residual + explained together --
    same convention fruresidual.eval.enrichment.category_enrichment uses).
    """
    if not residual_body_ids <= background_body_ids:
        raise ValueError("residual_body_ids must be a subset of background_body_ids")

    filtered = filter_by_min_count(partner_profile_long, min_neurons)
    n_residual = len(residual_body_ids)
    n_background = len(background_body_ids)

    rows = []
    for ptype, group in filtered.groupby("partner_type"):
        connected = set(group["bodyId"])
        a = len(connected & residual_body_ids)
        b = n_residual - a
        c = len(connected & background_body_ids) - a
        d = (n_background - n_residual) - c
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        # Haldane-Anscombe correction on the odds ratio for the same
        # zero-cell-stability reason as fruresidual.eval.enrichment.
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        rows.append(
            {
                "category": ptype,
                "n_in_subset": a,
                "n_in_background": a + c,
                "pct_of_subset": round(100 * a / n_residual, 1) if n_residual else 0.0,
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
