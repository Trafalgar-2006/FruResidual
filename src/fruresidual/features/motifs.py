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

import numpy as np
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


def partner_type_enrichment_degree_adjusted(
    partner_profile_long: pd.DataFrame,
    residual_body_ids: set[int],
    background_body_ids: set[int],
    min_neurons: int = 10,
    alpha: float = 0.05,
    n_degree_strata: int = 4,
    min_odds_ratio: float = 2.0,
) -> pd.DataFrame:
    """Degree-adjusted version of partner_type_enrichment.

    The marginal Fisher test in partner_type_enrichment cannot separate
    "residual neurons preferentially connect to type X" from "residual
    neurons are more highly connected overall, so they touch everything
    more" -- a confound measured as real in cb_intrinsic (residual mean
    degree 399 vs 322, p=0.0075) and vnc_intrinsic (518 vs 285,
    p=1.7e-6).

    Method: Cochran-Mantel-Haenszel. Neurons are binned into
    `n_degree_strata` quantile bins of partner-type degree, a 2x2
    [residual? x connects-to-type?] table is built per bin, and the
    association is tested conditional on those bins -- i.e. comparing
    residual against explained neurons of COMPARABLE connectivity
    intensity, never across degree levels.

    CMH rather than logistic regression on purpose. A per-type
    `is_residual ~ connects + log(degree)` logit was tried first and is
    unusable here: many partner types are touched by ~all residual and ~no
    explained neurons within a stratum (perfect separation), where an
    unpenalized fit will not converge and a weakly-penalized one drives
    the log-likelihood to zero, manufacturing p-values near 0 and odds
    ratios above 1e5. That inverted the method's purpose -- it reported
    MORE "significant" motifs than the uncorrected test (353 vs 68 on
    ascending_neuron). CMH is conditional, stays finite on sparse tables,
    and degrades to "not significant" rather than to false confidence.
    """
    from statsmodels.stats.contingency_tables import StratifiedTable

    if not residual_body_ids <= background_body_ids:
        raise ValueError("residual_body_ids must be a subset of background_body_ids")

    filtered = filter_by_min_count(partner_profile_long, min_neurons)
    degree = filtered.groupby("bodyId")["partner_type"].nunique()

    body_ids = sorted(background_body_ids)
    is_residual = np.array([b in residual_body_ids for b in body_ids])
    deg = degree.reindex(body_ids).fillna(0).to_numpy(dtype=float)

    # Quantile bins; duplicates='drop' collapses ties so a degenerate
    # degree distribution yields fewer (not broken) strata.
    try:
        bins = pd.qcut(deg, q=n_degree_strata, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        bins = np.zeros(len(deg), dtype=int)
    bins = np.asarray(bins)
    bins = np.where(pd.isna(bins), 0, bins).astype(int)

    rows = []
    for ptype, group in filtered.groupby("partner_type"):
        connected = set(group["bodyId"])
        connects = np.array([b in connected for b in body_ids])
        n_in_subset = int((connects & is_residual).sum())

        tables = []
        for b in np.unique(bins):
            m = bins == b
            a = int((connects & is_residual & m).sum())
            bb = int((~connects & is_residual & m).sum())
            c = int((connects & ~is_residual & m).sum())
            d = int((~connects & ~is_residual & m).sum())
            # A bin with an empty row or column margin carries no
            # information about the association; including it only adds
            # zeros to the CMH sums.
            if (a + bb) == 0 or (c + d) == 0 or (a + c) == 0 or (bb + d) == 0:
                continue
            tables.append([[a, bb], [c, d]])

        if not tables:
            rows.append({
                "category": ptype, "n_in_subset": n_in_subset,
                "n_in_background": int(connects.sum()),
                "odds_ratio": float("nan"), "p_value": float("nan"),
                "n_degree_strata_used": 0, "converged": False,
            })
            continue

        try:
            st = StratifiedTable(np.asarray(tables, dtype=float).transpose(1, 2, 0))
            pval = float(st.test_null_odds().pvalue)
            # Mantel-Haenszel pooled odds ratio with a Haldane-Anscombe
            # +0.5 per cell -- same correction fruresidual.eval.enrichment
            # applies, for the same reason. Without it a partner type
            # touched by every residual and no explained neuron in a
            # stratum makes the denominator exactly zero, so the OR comes
            # back as inf and the type gets discarded as "non-finite" --
            # silently throwing away the strongest signals.
            arr = np.asarray(tables, dtype=float) + 0.5
            a, b = arr[:, 0, 0], arr[:, 0, 1]
            c, d = arr[:, 1, 0], arr[:, 1, 1]
            n = a + b + c + d
            odds = float(np.sum(a * d / n) / np.sum(b * c / n))
            converged = bool(np.isfinite(pval) and np.isfinite(odds))
        except Exception:
            pval, odds, converged = float("nan"), float("nan"), False

        rows.append({
            "category": ptype, "n_in_subset": n_in_subset,
            "n_in_background": int(connects.sum()),
            "odds_ratio": odds, "p_value": pval,
            "n_degree_strata_used": len(tables), "converged": converged,
        })

    result = pd.DataFrame(rows)
    if not len(result):
        result["p_adj"] = []
        result["significant"] = []
        return result

    # BH correction over converged fits only -- including NaN p-values
    # would silently inflate the correction denominator.
    result["p_adj"] = float("nan")
    ok = result["converged"] & result["p_value"].notna()
    if ok.any():
        result.loc[ok, "p_adj"] = multipletests(
            result.loc[ok, "p_value"], alpha=alpha, method="fdr_bh"
        )[1]
    # Same significance bar as the marginal test: BH-adjusted p plus a
    # real effect size, so the two are directly comparable.
    result["significant"] = ok & (result["p_adj"] < alpha) & (result["odds_ratio"] >= min_odds_ratio)
    return result.sort_values("p_value", na_position="last").reset_index(drop=True)


def _cmh_vectorized(
    connects: np.ndarray, is_residual: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cochran-Mantel-Haenszel across ALL partner types simultaneously.

    `connects` is (n_neurons, n_types) boolean. Returns (chi2, odds_ratio)
    per type.

    This exists purely for speed, and the speed is what makes the
    permutation null possible at all: the per-type loop in
    partner_type_enrichment_degree_adjusted takes ~1 minute on
    cb_intrinsic's 4,860 types, so 1,000 shuffles would be ~16 hours.
    Recast as matrix operations over the neuron axis it is ~25ms, because
    every type shares the same degree bins and residual mask.

    Same statistic as statsmodels' StratifiedTable (continuity-corrected
    CMH) and the same Haldane-Anscombe +0.5 pooled odds ratio, verified
    against it in test_vectorized_cmh_matches_statsmodels.
    """
    res = is_residual[:, None]
    num = np.zeros(connects.shape[1])
    var = np.zeros(connects.shape[1])
    or_num = np.zeros(connects.shape[1])
    or_den = np.zeros(connects.shape[1])

    for b in np.unique(bins):
        m = bins == b
        cm, rm = connects[m], res[m]
        a = (cm & rm).sum(axis=0).astype(float)
        bb = (~cm & rm).sum(axis=0).astype(float)
        c = (cm & ~rm).sum(axis=0).astype(float)
        d = (~cm & ~rm).sum(axis=0).astype(float)
        n = a + bb + c + d
        r1, r2, c1, c2 = a + bb, c + d, a + c, bb + d
        # Bins with an empty margin carry no information; zeroing them
        # out is equivalent to the per-type code skipping the table.
        ok = (n > 1) & (r1 > 0) & (r2 > 0) & (c1 > 0) & (c2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(ok, r1 * c1 / n, 0.0)
            v = np.where(ok, r1 * r2 * c1 * c2 / (n * n * (n - 1)), 0.0)
        num += np.where(ok, a, 0.0) - e
        var += v
        ah, bh, ch, dh = a + 0.5, bb + 0.5, c + 0.5, d + 0.5
        nh = ah + bh + ch + dh
        or_num += np.where(ok, ah * dh / nh, 0.0)
        or_den += np.where(ok, bh * ch / nh, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # No continuity correction: statsmodels' test_null_odds() defaults
        # to correction=False, and the reported results come from that
        # call. Applying the usual -0.5 here would make the null measure a
        # slightly different (more conservative) statistic than the thing
        # it is validating -- caught by test_vectorized_cmh_matches_statsmodels.
        chi2 = np.where(var > 0, num**2 / var, np.nan)
        odds = np.where(or_den > 0, or_num / or_den, np.nan)
    return chi2, odds


def motif_permutation_null(
    partner_profile_long: pd.DataFrame,
    residual_body_ids: set[int],
    background_body_ids: set[int],
    min_neurons: int = 10,
    alpha: float = 0.05,
    min_odds_ratio: float = 2.0,
    n_degree_strata: int = 4,
    n_permutations: int = 1000,
    random_state: int = 0,
) -> dict:
    """Empirical null for the NUMBER of significant degree-adjusted motifs.

    partner_type_enrichment_degree_adjusted reports BH-corrected
    p-values, which control the false-discovery rate under assumptions
    about the test statistics' joint behaviour. Those assumptions are
    shaky here: partner types are heavily correlated (neurons that touch
    one type tend to touch related ones), so the effective number of
    independent tests is far below the nominal count and BH may be
    optimistic.

    This asks the question directly. Hold the connectivity matrix and the
    degree bins FIXED and shuffle which neurons carry the residual label,
    preserving the residual set's size exactly. Each shuffle yields a
    count of "significant" motifs under identical machinery. The observed
    count is then read against that distribution.

    Returns the observed count, the null distribution's summary, and an
    empirical p-value -- plus `expected_false_positives`, which is the
    null mean and is the number that actually matters for reading the
    shortlist.
    """
    from scipy.stats import chi2 as chi2_dist

    if not residual_body_ids <= background_body_ids:
        raise ValueError("residual_body_ids must be a subset of background_body_ids")

    filtered = filter_by_min_count(partner_profile_long, min_neurons)
    body_ids = sorted(background_body_ids)
    idx = {b: i for i, b in enumerate(body_ids)}
    types = sorted(filtered["partner_type"].unique())
    tidx = {t: i for i, t in enumerate(types)}

    connects = np.zeros((len(body_ids), len(types)), dtype=bool)
    rows = filtered["bodyId"].map(idx).to_numpy()
    cols = filtered["partner_type"].map(tidx).to_numpy()
    keep = ~pd.isna(rows)
    connects[rows[keep].astype(int), cols[keep].astype(int)] = True

    degree = connects.sum(axis=1).astype(float)
    try:
        bins = pd.qcut(degree, q=n_degree_strata, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        bins = np.zeros(len(degree), dtype=int)
    bins = np.where(pd.isna(np.asarray(bins)), 0, np.asarray(bins)).astype(int)

    def _n_significant(mask: np.ndarray) -> int:
        chi2, odds = _cmh_vectorized(connects, mask, bins)
        ok = np.isfinite(chi2) & np.isfinite(odds)
        if not ok.any():
            return 0
        p = chi2_dist.sf(chi2[ok], df=1)
        p_adj = multipletests(p, alpha=alpha, method="fdr_bh")[1]
        return int(np.sum((p_adj < alpha) & (odds[ok] >= min_odds_ratio)))

    observed_mask = np.array([b in residual_body_ids for b in body_ids])
    observed = _n_significant(observed_mask)

    rng = np.random.default_rng(random_state)
    n_res = int(observed_mask.sum())
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = np.zeros(len(body_ids), dtype=bool)
        shuffled[rng.choice(len(body_ids), size=n_res, replace=False)] = True
        null[i] = _n_significant(shuffled)

    n_ge = int(np.sum(null >= observed))
    return {
        "n_types_tested": len(types),
        "observed_n_significant": observed,
        "n_permutations": n_permutations,
        "expected_false_positives": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_p95": float(np.percentile(null, 95)),
        "null_max": float(np.max(null)),
        "n_null_ge_observed": n_ge,
        "empirical_p_value": float((n_ge + 1) / (n_permutations + 1)),
        "excess_over_null": observed - float(np.mean(null)),
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
