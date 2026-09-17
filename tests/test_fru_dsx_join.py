import pandas as pd

from fruresidual.features.fru_dsx_join import build_dimorphism_crosstab, phase0_gate_summary


def _small_toy_df():
    return pd.DataFrame(
        {
            "dimorphism": ["sexually dimorphic"] * 6 + ["male-specific"] * 4 + [None] * 5,
            "fruDsx": ["fru_high", "fru_low", None, None, None, None,
                       "dsx_high", "dsx_high", None, None,
                       None, None, None, None, None],
            "superclass": ["ol_intrinsic"] * 5 + ["cb_intrinsic"] + ["ol_intrinsic"] * 4 + ["x"] * 5,
        }
    )


def _large_toy_df():
    """200 dimorphic neurons split evenly across two superclasses, with
    fru/dsx positivity rates deliberately swapped (10% vs 90%) so the
    residual (fru/dsx-negative) is heavily concentrated in ol_intrinsic --
    a signal large enough for Fisher's exact test to actually detect,
    unlike the 15-row toy dataset above.
    """
    n_per_class = 100
    ol_fru_dsx = ["fru_high"] * 10 + [None] * 90
    cb_fru_dsx = ["fru_high"] * 90 + [None] * 10
    return pd.DataFrame(
        {
            "dimorphism": ["sexually dimorphic"] * (2 * n_per_class),
            "superclass": ["ol_intrinsic"] * n_per_class + ["cb_intrinsic"] * n_per_class,
            "fruDsx": ol_fru_dsx + cb_fru_dsx,
        }
    )


def test_crosstab_counts_positive_and_negative():
    df = _small_toy_df()
    ct = build_dimorphism_crosstab(df, "any_positive")
    assert ct.loc["sexually dimorphic", True] == 2
    assert ct.loc["sexually dimorphic", False] == 4
    assert ct.loc["male-specific", True] == 2
    assert ct.loc["male-specific", False] == 2


def test_high_confidence_only_is_stricter():
    df = _small_toy_df()
    any_pos = phase0_gate_summary(df, "any_positive")
    high_conf = phase0_gate_summary(df, "high_confidence_only")
    assert high_conf["n_residual_unexplained"] >= any_pos["n_residual_unexplained"]


def test_gate_detects_real_enrichment():
    df = _large_toy_df()
    summary = phase0_gate_summary(df, "any_positive")
    assert summary["structured"] is True
    assert summary["verdict"] == "ALIVE"
    enriched_names = {row["category"] for row in summary["enriched_superclasses"]}
    assert "ol_intrinsic" in enriched_names
    assert "cb_intrinsic" not in enriched_names  # cb is depleted in the residual, not enriched


def test_gate_rejects_unstructured_residual():
    """A residual spread proportionally to the background (no real signal)
    should NOT be flagged as structured, even if it's large."""
    n_per_class = 100
    # Same ~50% positivity rate in both superclasses -> residual should
    # land roughly proportionally in both, no enrichment either way.
    ol_fru_dsx = ["fru_high"] * 50 + [None] * 50
    cb_fru_dsx = ["fru_high"] * 50 + [None] * 50
    df = pd.DataFrame(
        {
            "dimorphism": ["sexually dimorphic"] * (2 * n_per_class),
            "superclass": ["ol_intrinsic"] * n_per_class + ["cb_intrinsic"] * n_per_class,
            "fruDsx": ol_fru_dsx + cb_fru_dsx,
        }
    )
    summary = phase0_gate_summary(df, "any_positive")
    assert summary["structured"] is False
    assert summary["verdict"] == "INCONCLUSIVE"
