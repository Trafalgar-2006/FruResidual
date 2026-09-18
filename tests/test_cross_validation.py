import pandas as pd
import pytest

from fruresidual.features.cross_validation import compute_concordance, match_banc_to_malecns


def _toy_banc_df():
    return pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6, 7],
            "malecns_match": [100, 100, 200, 300, 300, 300, 400],  # 100: 2x, 300: 3x, 200/400: 1x
            "cell_type": ["A"] * 7,
            "sexually_dimorphic": ["dimorphic", "dimorphic", "isomorphic", "dimorphic", "isomorphic", "isomorphic", "dimorphic"],
            "proofread": [True] * 7,
        }
    )


def test_match_banc_to_malecns_keeps_only_unambiguous_matches():
    banc_df = _toy_banc_df()
    matched = match_banc_to_malecns(banc_df, malecns_body_ids={100, 200, 300, 400})
    # 100 (2x) and 300 (3x) are ambiguous -- excluded. 200 and 400 (1x each) kept.
    assert set(matched["bodyId"]) == {200, 400}


def test_match_banc_to_malecns_filters_to_given_population():
    banc_df = _toy_banc_df()
    matched = match_banc_to_malecns(banc_df, malecns_body_ids={200})  # 400 not in population
    assert set(matched["bodyId"]) == {200}


def _toy_matched_and_labels():
    # 20 residual (fru/dsx-negative) MaleCNS neurons, 20 explained.
    # BANC confirms dimorphism for 15/20 residual (75%) but only 5/20
    # explained (25%) -- a clear, testable difference.
    body_ids = list(range(40))
    residual_ids = set(range(20))
    banc_dimorphic_flags = [i < 15 for i in range(20)] + [i < 5 for i in range(20)]
    matched = pd.DataFrame(
        {
            "bodyId": body_ids,
            "root_id": [1000 + i for i in body_ids],
            "cell_type": ["X"] * 40,
            "sexually_dimorphic": ["dimorphic" if f else "isomorphic" for f in banc_dimorphic_flags],
            "proofread": [True] * 40,
        }
    )
    # All 40 are MaleCNS-side dimorphic (residual + explained are both
    # subsets of the dimorphic population, per fru_dsx_join's convention).
    malecns_dimorphism = pd.Series(["sexually dimorphic"] * 40, index=body_ids)
    return matched, malecns_dimorphism, residual_ids


def test_compute_concordance_jaccard_and_rates():
    matched, malecns_dimorphism, residual_ids = _toy_matched_and_labels()
    result = compute_concordance(matched, malecns_dimorphism, residual_ids)

    assert result["n_matched_total"] == 40
    assert result["n_residual_matched"] == 20
    assert result["n_explained_matched"] == 20
    assert result["pct_residual_confirmed_dimorphic_by_banc"] == pytest.approx(75.0)
    assert result["pct_explained_confirmed_dimorphic_by_banc"] == pytest.approx(25.0)
    # all 40 are malecns_dimorphic; banc_dimorphic = 15+5=20 -> union=40, intersection=20
    assert result["jaccard_dimorphic_overlap"] == pytest.approx(20 / 40)
    # recall: of the 40 malecns_dimorphic neurons, 20 confirmed by BANC
    assert result["recall_malecns_dimorphic_confirmed_by_banc"] == pytest.approx(20 / 40)
    # precision: of BANC's 20 dimorphic calls here, all 20 are also malecns_dimorphic (all 40 rows are)
    assert result["precision_banc_dimorphic_matches_malecns"] == pytest.approx(1.0)


def test_compute_concordance_recall_and_precision_diverge_with_extra_banc_only_calls():
    # Add rows where BANC calls dimorphic but MaleCNS does NOT (isomorphic
    # on the MaleCNS side) -- precision should drop while recall doesn't.
    matched, malecns_dimorphism, residual_ids = _toy_matched_and_labels()
    extra = pd.DataFrame(
        {
            "bodyId": list(range(40, 60)),
            "root_id": list(range(2000, 2020)),
            "cell_type": ["Y"] * 20,
            "sexually_dimorphic": ["dimorphic"] * 20,
            "proofread": [True] * 20,
        }
    )
    matched_extended = pd.concat([matched, extra], ignore_index=True)
    extra_labels = pd.Series([None] * 20, index=range(40, 60))  # isomorphic on MaleCNS side
    malecns_dimorphism_extended = pd.concat([malecns_dimorphism, extra_labels])

    result = compute_concordance(matched_extended, malecns_dimorphism_extended, residual_ids)
    assert result["recall_malecns_dimorphic_confirmed_by_banc"] == pytest.approx(20 / 40)  # unchanged
    assert result["precision_banc_dimorphic_matches_malecns"] < 1.0  # now diluted by the 20 extra BANC-only calls


def test_compute_concordance_detects_significant_difference():
    matched, malecns_dimorphism, residual_ids = _toy_matched_and_labels()
    result = compute_concordance(matched, malecns_dimorphism, residual_ids)
    assert result["residual_vs_explained_p_value"] < 0.05


def test_compute_concordance_no_difference_gives_high_p_value():
    body_ids = list(range(40))
    residual_ids = set(range(20))
    # identical 50% confirmation rate in both groups -- no real signal
    banc_dimorphic_flags = [i % 2 == 0 for i in range(40)]
    matched = pd.DataFrame(
        {
            "bodyId": body_ids,
            "root_id": [1000 + i for i in body_ids],
            "cell_type": ["X"] * 40,
            "sexually_dimorphic": ["dimorphic" if f else "isomorphic" for f in banc_dimorphic_flags],
            "proofread": [True] * 40,
        }
    )
    malecns_dimorphism = pd.Series(["sexually dimorphic"] * 40, index=body_ids)
    result = compute_concordance(matched, malecns_dimorphism, residual_ids)
    assert result["residual_vs_explained_p_value"] > 0.05
