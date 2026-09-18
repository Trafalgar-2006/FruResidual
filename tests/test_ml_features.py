import pandas as pd
import pytest

from fruresidual.features.ml_features import build_connectivity_summary_features, build_features


def _toy_population():
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "superclass": ["ol_intrinsic", "cb_intrinsic", "ol_intrinsic"],
            "subclass": ["a", "b", "a"],
            "somaSide": ["L", "R", "L"],
            "consensus_nt": ["acetylcholine", "gaba", "acetylcholine"],
            "dimorphism": ["sexually dimorphic", None, None],
            "flywireType": ["T1", "T2", "T1"],
            "fruDsx": ["fru_high", None, None],
        }
    )


def _toy_profiles():
    return pd.DataFrame(
        {
            "bodyId": [1, 1, 2, 3],
            "partner_superclass": ["ol_intrinsic", "cb_intrinsic", "cb_intrinsic", "ol_intrinsic"],
            "weight": [10, 5, 20, 3],
        }
    )


def test_connectivity_summary_computes_degree_and_entropy():
    summary = build_connectivity_summary_features(_toy_profiles())
    row1 = summary[summary["bodyId"] == 1].iloc[0]
    assert row1["total_connectivity_weight"] == 15
    assert row1["n_partner_superclasses"] == 2
    assert row1["connectivity_entropy"] > 0  # mixed distribution has positive entropy

    row3 = summary[summary["bodyId"] == 3].iloc[0]
    assert row3["connectivity_entropy"] == pytest.approx(0.0, abs=1e-9)  # single partner class


def test_build_features_excludes_leakage_columns_by_default():
    X = build_features(_toy_population(), _toy_profiles(), include_fru_dsx=False)
    for leaky in ("dimorphism", "flywireType", "fruDsx"):
        assert leaky not in X.columns


def test_build_features_ablation_flag_adds_fru_dsx_feature():
    without = build_features(_toy_population(), _toy_profiles(), include_fru_dsx=False)
    with_fd = build_features(_toy_population(), _toy_profiles(), include_fru_dsx=True)
    assert "fru_dsx_positive" not in without.columns
    assert "fru_dsx_positive" in with_fd.columns
    assert with_fd["fru_dsx_positive"].tolist() == [1, 0, 0]


def test_build_features_row_count_matches_population():
    X = build_features(_toy_population(), _toy_profiles())
    assert len(X) == 3
