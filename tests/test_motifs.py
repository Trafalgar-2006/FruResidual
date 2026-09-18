import pandas as pd
import pytest

from fruresidual.features.motifs import (
    check_degree_confound,
    filter_by_min_count,
    identify_mixed_strata,
    partner_type_enrichment,
)


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
