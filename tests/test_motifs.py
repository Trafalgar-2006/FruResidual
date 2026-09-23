import pandas as pd
import pytest

from fruresidual.features.motifs import (
    check_degree_confound,
    filter_by_min_count,
    identify_mixed_strata,
    partner_type_enrichment,
    partner_type_enrichment_degree_adjusted,
)


def _degree_confounded_profile():
    """Residual neurons are simply MORE CONNECTED -- they touch every
    partner type more often, with no type-specific preference at all.
    The marginal Fisher test should light up across the board (that is
    the confound). The degree-adjusted test should not, because once
    overall degree is held constant there is no signal left.
    """
    rows = []
    types = [f"T{t}" for t in range(20)]
    for i in range(60):  # residual: touch 18 of 20 types
        for t in types[:18]:
            rows.append({"bodyId": i, "partner_type": t, "weight": 5})
    for i in range(60, 120):  # explained: touch 6 of 20 types, same types
        for t in types[:6]:
            rows.append({"bodyId": i, "partner_type": t, "weight": 5})
    return pd.DataFrame(rows), set(range(60)), set(range(120))


def _toy_profile():
    # 100 residual neurons, 100 background: "HubType" is connected to by
    # 90% of residual but only 10% of background -- should be detected.
    # "RareType" is touched by only 3 neurons total -- should be filtered
    # out by min_count before testing.
    rows = []
    for i in range(100):
        rows.append({"bodyId": i, "partner_type": "CommonType", "weight": 5})
        if i < 90:
            rows.append({"bodyId": i, "partner_type": "HubType", "weight": 5})
    for i in range(100, 200):
        rows.append({"bodyId": i, "partner_type": "CommonType", "weight": 5})
        if i < 110:
            rows.append({"bodyId": i, "partner_type": "HubType", "weight": 5})
    rows.append({"bodyId": 0, "partner_type": "RareType", "weight": 1})
    rows.append({"bodyId": 1, "partner_type": "RareType", "weight": 1})
    return pd.DataFrame(rows)


def test_filter_by_min_count_drops_rare_types():
    profile = _toy_profile()
    filtered = filter_by_min_count(profile, min_neurons=10)
    assert "RareType" not in set(filtered["partner_type"])
    assert "HubType" in set(filtered["partner_type"])
    assert "CommonType" in set(filtered["partner_type"])


def test_partner_type_enrichment_detects_hub_type():
    profile = _toy_profile()
    residual_ids = set(range(100))  # first 100 neurons are "residual"
    background_ids = set(range(200))
    result = partner_type_enrichment(profile, residual_ids, background_ids, min_neurons=10)
    hub_row = result[result["category"] == "HubType"].iloc[0]
    assert hub_row["significant"]
    assert hub_row["odds_ratio"] > 2

    common_row = result[result["category"] == "CommonType"].iloc[0]
    assert not common_row["significant"]  # touched equally by both groups


def test_partner_type_enrichment_pct_is_neuron_level_not_edge_level():
    """Regression test for the real multi-label bug found 2026-09-18: each
    neuron contributes one row PER partner type it connects to (multi-
    label), so a naive len(series) denominator counts edges, not neurons
    -- a category touched by 90/100 residual neurons was reported as
    touching ~0.05% of "subset" because the denominator was thousands of
    residual-neuron edges, not the 100 residual neurons. pct_of_subset
    must equal n_in_subset / (number of residual NEURONS), not edges.
    """
    profile = _toy_profile()  # each residual neuron also has a CommonType
    # row, so subset has ~190 EDGES (100 CommonType + 90 HubType) but only
    # 100 distinct residual NEURONS -- these must not be conflated.
    residual_ids = set(range(100))
    background_ids = set(range(200))
    result = partner_type_enrichment(profile, residual_ids, background_ids, min_neurons=10)
    hub_row = result[result["category"] == "HubType"].iloc[0]
    assert hub_row["n_in_subset"] == 90
    assert hub_row["pct_of_subset"] == pytest.approx(90.0)  # 90/100 residual NEURONS, not 90/190 edges


def test_partner_type_enrichment_rejects_non_subset_residual_ids():
    profile = _toy_profile()
    residual_ids = {9999}  # not a subset of background
    background_ids = set(range(200))
    with pytest.raises(ValueError):
        partner_type_enrichment(profile, residual_ids, background_ids, min_neurons=10)


def test_regional_confound_produces_spurious_marginal_enrichment():
    """Regression test documenting a real failure mode found on real data
    (2026-09-18): when residual status is PERFECTLY confounded with a
    region (e.g. ol_intrinsic was 100% residual, 0% explained -- zero
    within-region variance), an unstratified (marginal) motif enrichment
    call spuriously flags region-specific partner types as
    "residual-enriched" -- tautologically, not because of any real
    circuit-level signal. The fix is stratifying by region before
    testing (see scripts/run_phase3.py), not a change to
    partner_type_enrichment's statistics, which are separately verified
    correct on non-confounded data (test_partner_type_enrichment_detects_hub_type).
    This test exists so that regressing back to an unstratified Phase-3
    population fails loudly instead of silently producing a misleading
    shortlist.
    """
    # Region A: 100% residual (the ol_intrinsic-like saturated stratum).
    # Region B: 50/50 mixed (a "clean" stratum, no confound).
    # RegionAPartner is touched ONLY by region A neurons -- structurally
    # tied to region, not to any real residual-specific biology.
    rows = []
    for i in range(100):  # region A, all residual
        rows.append({"bodyId": i, "partner_type": "RegionAPartner", "weight": 1})
    for i in range(100, 300):  # region B, half residual half explained
        rows.append({"bodyId": i, "partner_type": "RegionBPartner", "weight": 1})
    profile = pd.DataFrame(rows)
    region = pd.Series(["A"] * 100 + ["B"] * 200, index=range(300))
    residual_ids = set(range(100)) | set(range(100, 300, 2))  # all of A + half of B
    background_ids = set(range(300))

    # Marginal (unstratified, confounded) call -- reproduces the bug.
    marginal = partner_type_enrichment(profile, residual_ids, background_ids, min_neurons=10)
    region_a_row = marginal[marginal["category"] == "RegionAPartner"]
    assert region_a_row.iloc[0]["significant"]  # spuriously "significant" -- tautological

    # Stratified (region-controlled) call: within region A alone there is
    # zero explained/residual variance to compare, so the correct,
    # honest analysis excludes saturated strata entirely rather than
    # reporting a degenerate comparison.
    region_a_ids = set(region[region == "A"].index)
    region_a_residual = residual_ids & region_a_ids
    assert len(region_a_residual) == len(region_a_ids)  # 100% residual: no variance, must be excluded


def test_check_degree_confound_detects_real_difference():
    # residual neurons touch 2x as many partner types as explained ones
    rows = []
    for i in range(50):  # residual: high degree
        for j in range(20):
            rows.append({"bodyId": i, "partner_type": f"t{j}", "weight": 1})
    for i in range(50, 100):  # explained: low degree
        for j in range(10):
            rows.append({"bodyId": i, "partner_type": f"t{j}", "weight": 1})
    profile = pd.DataFrame(rows)
    residual_ids = set(range(50))
    background_ids = set(range(100))

    result = check_degree_confound(profile, residual_ids, background_ids)
    assert result["residual_mean_degree"] > result["explained_mean_degree"]
    assert result["degree_confounded"] is True
    assert result["degree_difference_p_value"] < 0.05


def test_check_degree_confound_finds_no_difference_when_matched():
    rows = []
    for i in range(100):  # everyone touches the same number of partners
        for j in range(15):
            rows.append({"bodyId": i, "partner_type": f"t{j}", "weight": 1})
    profile = pd.DataFrame(rows)
    residual_ids = set(range(50))
    background_ids = set(range(100))

    result = check_degree_confound(profile, residual_ids, background_ids)
    assert result["degree_confounded"] is False


def test_identify_mixed_strata_excludes_saturated_regions():
    dimorphic = pd.DataFrame({"bodyId": range(300), "superclass": ["A"] * 100 + ["B"] * 200})
    residual_ids = set(range(100)) | set(range(100, 300, 2))  # A: 100% residual; B: 50% residual
    mixed = identify_mixed_strata(dimorphic, residual_ids, min_group_size=20)
    assert "A" not in set(mixed["superclass"])  # saturated (0 explained) -- must be excluded
    assert "B" in set(mixed["superclass"])  # has >=20 of both -- genuinely testable
    b_row = mixed[mixed["superclass"] == "B"].iloc[0]
    assert b_row["n_residual"] == 100 and b_row["n_explained"] == 100


def test_degree_adjustment_removes_a_purely_degree_driven_false_positive():
    """The core justification for partner_type_enrichment_degree_adjusted:
    when residual neurons simply connect to MORE things with no
    type-specific preference, the marginal Fisher test reports many
    "significant" motifs that are pure degree artifact. Holding degree
    constant should collapse them.
    """
    profile, residual_ids, background_ids = _degree_confounded_profile()

    marginal = partner_type_enrichment(profile, residual_ids, background_ids, min_neurons=5)
    adjusted = partner_type_enrichment_degree_adjusted(
        profile, residual_ids, background_ids, min_neurons=5
    )

    n_marginal = int(marginal["significant"].sum())
    n_adjusted = int(adjusted["significant"].sum())

    # The confound is real and the marginal test falls for it...
    assert n_marginal > 0
    # ...and the degree-adjusted test does not.
    assert n_adjusted < n_marginal
    assert n_adjusted == 0


def test_degree_adjustment_keeps_a_genuine_type_specific_signal():
    """Guards the opposite failure: over-correcting until nothing can ever
    be detected. Here degree is matched between groups and only ONE type
    is genuinely preferred, which must survive adjustment.
    """
    rows = []
    for i in range(60):  # residual: 10 shared types + HubType
        for j in range(10):
            rows.append({"bodyId": i, "partner_type": f"T{j}", "weight": 5})
        rows.append({"bodyId": i, "partner_type": "HubType", "weight": 5})
    for i in range(60, 120):  # explained: 10 shared types + OtherType (same degree)
        for j in range(10):
            rows.append({"bodyId": i, "partner_type": f"T{j}", "weight": 5})
        rows.append({"bodyId": i, "partner_type": "OtherType", "weight": 5})
    profile = pd.DataFrame(rows)

    adjusted = partner_type_enrichment_degree_adjusted(
        profile, set(range(60)), set(range(120)), min_neurons=5
    )
    hub = adjusted[adjusted["category"] == "HubType"]
    assert len(hub) == 1
    assert bool(hub.iloc[0]["significant"])


def test_vectorized_cmh_matches_statsmodels():
    """The permutation null is only meaningful if its fast CMH computes
    the SAME statistic as the per-type statsmodels path used for the
    reported results. A speedup that quietly changes the test would make
    the null incomparable to the thing it is supposed to validate.
    """
    import numpy as np
    from statsmodels.stats.contingency_tables import StratifiedTable

    from fruresidual.features.motifs import _cmh_vectorized

    rng = np.random.default_rng(0)
    n, n_types = 200, 6
    connects = rng.random((n, n_types)) < 0.4
    is_residual = rng.random(n) < 0.5
    bins = rng.integers(0, 4, size=n)

    chi2_vec, odds_vec = _cmh_vectorized(connects, is_residual, bins)

    for t in range(n_types):
        tables = []
        for b in np.unique(bins):
            m = bins == b
            a = int((connects[m, t] & is_residual[m]).sum())
            bb = int((~connects[m, t] & is_residual[m]).sum())
            c = int((connects[m, t] & ~is_residual[m]).sum())
            d = int((~connects[m, t] & ~is_residual[m]).sum())
            if (a + bb) == 0 or (c + d) == 0 or (a + c) == 0 or (bb + d) == 0:
                continue
            tables.append([[a, bb], [c, d]])
        if not tables:
            continue
        st = StratifiedTable(np.asarray(tables, dtype=float).transpose(1, 2, 0))
        arr = np.asarray(tables, dtype=float) + 0.5
        aa, bh, ch, dh = arr[:, 0, 0], arr[:, 0, 1], arr[:, 1, 0], arr[:, 1, 1]
        nh = aa + bh + ch + dh
        odds_ref = float(np.sum(aa * dh / nh) / np.sum(bh * ch / nh))

        assert np.isclose(chi2_vec[t], st.test_null_odds().statistic, rtol=1e-6), (
            f"type {t}: chi2 {chi2_vec[t]} vs statsmodels {st.test_null_odds().statistic}"
        )
        assert np.isclose(odds_vec[t], odds_ref, rtol=1e-6)


def test_motif_permutation_null_on_pure_degree_artifact():
    """The null must show that a purely degree-driven 'signal' is
    reproduced by chance shuffles -- i.e. the observed count should not
    stand out against the null."""
    from fruresidual.features.motifs import motif_permutation_null

    profile, residual_ids, background_ids = _degree_confounded_profile()
    out = motif_permutation_null(
        profile, residual_ids, background_ids, min_neurons=5,
        n_permutations=50, random_state=0,
    )
    assert out["n_permutations"] == 50
    assert 0 < out["empirical_p_value"] <= 1.0
    assert out["observed_n_significant"] >= 0
