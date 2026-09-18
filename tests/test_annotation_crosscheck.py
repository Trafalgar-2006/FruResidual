import pandas as pd

from fruresidual.features.annotation_crosscheck import (
    build_flywire_type_lookup,
    crosscheck_residual,
    flywire_coverage_by_super_class,
    independence_diagnostic,
    match_dimorphic_to_flywire,
)


def _fw(rows):
    return pd.DataFrame(rows, columns=["cell_type", "super_class", "fru_dsx"])


def test_type_lookup_uses_any_positive_rule():
    # "any" is deliberate: it is the rule most likely to FIND a fru/dsx
    # call on the residual, so it biases against this analysis's own
    # conclusion rather than toward it.
    fw = _fw([("T1", "central", None), ("T1", "central", "fru"), ("T2", "central", None)])
    lookup = build_flywire_type_lookup(fw)
    assert bool(lookup.loc["T1", "fw_fru_dsx_any"]) is True
    assert bool(lookup.loc["T2", "fw_fru_dsx_any"]) is False


def test_coverage_by_super_class_flags_blind_spots():
    fw = _fw([("A", "central", "fru"), ("B", "central", None),
              ("C", "optic", None), ("D", "optic", None)])
    cov = flywire_coverage_by_super_class(fw)
    assert cov.loc["central", "pct_positive"] == 50.0
    assert cov.loc["optic", "pct_positive"] == 0.0


def test_ambiguous_multi_type_matches_are_excluded():
    fw = _fw([("T1", "central", "fru")])
    lookup = build_flywire_type_lookup(fw)
    dim = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "flywireType": ["T1", "T1,T2", None],
        "has_fru_dsx": [True, True, True],
    })
    m = match_dimorphic_to_flywire(dim, lookup)
    assert list(m["bodyId"]) == [1]


def test_independence_diagnostic_detects_nested_annotation():
    """The decisive check. If the second annotation never fires where the
    first didn't, it is a subset of the first -- so "the residual is
    negative in both" is true by construction and proves nothing. This
    is the real situation with flywire_annotations' fru_dsx column, whose
    values were derived with reference to MaleCNS.
    """
    nested = pd.DataFrame({
        "has_fru_dsx": [True, True, True, False, False],
        "fw_fru_dsx_any": [True, True, False, False, False],  # never + where MaleCNS is -
    })
    d = independence_diagnostic(nested)
    assert d["n_malecns_neg_flywire_pos"] == 0
    assert d["flywire_labels_nested_within_malecns"] is True
    assert "NOT INDEPENDENT" in d["independence_verdict"]


def test_independence_diagnostic_accepts_genuinely_independent_source():
    independent = pd.DataFrame({
        "has_fru_dsx": [True, True, False, False],
        "fw_fru_dsx_any": [True, False, True, False],  # fires where MaleCNS did not
    })
    d = independence_diagnostic(independent)
    assert d["n_malecns_neg_flywire_pos"] == 1
    assert d["flywire_labels_nested_within_malecns"] is False
    assert "PARTIALLY INDEPENDENT" in d["independence_verdict"]


def test_crosscheck_computes_expected_count_under_the_null():
    matched = pd.DataFrame({
        "has_fru_dsx": [True] * 10 + [False] * 10,
        "fw_fru_dsx_any": [True] * 8 + [False] * 2 + [False] * 10,
        "fw_super_class": ["central"] * 20,
    })
    res = crosscheck_residual(matched)["overall"]
    assert res["n_residual"] == 10
    assert res["n_residual_flywire_positive"] == 0
    assert res["pct_explained_flywire_positive"] == 80.0
    # if the residual behaved like the explained set we'd expect 8 positives
    assert res["expected_residual_positive_at_explained_rate"] == 8.0
    assert res["p_residual_depleted"] < 0.05
