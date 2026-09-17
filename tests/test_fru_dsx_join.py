import pandas as pd

from fruresidual.features.fru_dsx_join import build_dimorphism_crosstab, phase0_gate_summary


def _toy_df():
    return pd.DataFrame(
        {
            "dimorphism": ["sexually dimorphic"] * 6 + ["male-specific"] * 4 + [None] * 5,
            "fruDsx": ["fru_high", "fru_low", None, None, None, None,
                       "dsx_high", "dsx_high", None, None,
                       None, None, None, None, None],
            "superclass": ["ol_intrinsic"] * 5 + ["cb_intrinsic"] + ["ol_intrinsic"] * 4 + ["x"] * 5,
        }
    )


def test_crosstab_counts_positive_and_negative():
    df = _toy_df()
    ct = build_dimorphism_crosstab(df, "any_positive")
    assert ct.loc["sexually dimorphic", True] == 2
    assert ct.loc["sexually dimorphic", False] == 4
    assert ct.loc["male-specific", True] == 2
    assert ct.loc["male-specific", False] == 2


def test_high_confidence_only_is_stricter():
    df = _toy_df()
    any_pos = phase0_gate_summary(df, "any_positive", concentration_threshold=0.5)
    high_conf = phase0_gate_summary(df, "high_confidence_only", concentration_threshold=0.5)
    assert high_conf["n_residual_unexplained"] >= any_pos["n_residual_unexplained"]


def test_gate_flags_structured_residual():
    df = _toy_df()
    summary = phase0_gate_summary(df, "any_positive", concentration_threshold=0.5)
    assert summary["structured"] is True
    assert summary["top_superclasses"]["ol_intrinsic"] >= 1
