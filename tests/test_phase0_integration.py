"""Integration test against the real downloaded MaleCNS data.

Skipped (not failed) when the data file isn't present -- e.g. a fresh
clone before running scripts/download_male_cns.py, or CI without the
~14MB download. Where the unit tests in test_fru_dsx_join.py use small
synthetic frames to test the logic in isolation, this test catches
regressions in the actual pipeline output against the real file: if the
data changes shape, the gate logic breaks, or the verdict flips, this is
what would tell us.
"""

import pytest

from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    load_male_cns_annotations,
    phase0_gate_summary,
)

cfg = load_config()
_data_path = cfg.data_raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

pytestmark = pytest.mark.skipif(
    not _data_path.exists(),
    reason=f"MaleCNS data not downloaded (expected at {_data_path}); "
    "run scripts/download_male_cns.py first",
)


@pytest.fixture(scope="module")
def real_traced_df():
    return load_male_cns_annotations(cfg, traced_only=True)


def test_expected_columns_present(real_traced_df):
    for col in ("dimorphism", "fruDsx", "superclass", "status", "type", "bodyId"):
        assert col in real_traced_df.columns


def test_traced_count_in_expected_range(real_traced_df):
    # Paper's published total CNS neuron count is 166,700; Traced-only
    # should be close to but not exceed that (some non-neuron/low-quality
    # entries get excluded by the Traced filter).
    assert 150_000 < len(real_traced_df) < 170_000


def test_dimorphism_categories_match_known_schema(real_traced_df):
    observed = set(real_traced_df["dimorphism"].dropna().unique())
    assert observed == set(DIMORPHIC_CATEGORIES)


def test_absolute_fru_dsx_counts_match_published_paper(real_traced_df):
    """Regression-pins the exact match discovered on 2026-09-17: paper
    reports 4,858 fru (2,804 high-confidence), 412 dsx (331 high-conf),
    258 co-expressing -- which only matches our data if co-expressing
    neurons are counted toward BOTH the fru and dsx totals.
    """
    fd = real_traced_df["fruDsx"]
    n_fru_only = fd.isin({"fru_high", "fru_low"}).sum()
    n_dsx_only = fd.isin({"dsx_high", "dsx_low"}).sum()
    n_coexpress = fd.isin({"coexpress_high", "coexpress_low"}).sum()
    n_fru_high_only = (fd == "fru_high").sum()
    n_dsx_high_only = (fd == "dsx_high").sum()
    n_coexpress_high = (fd == "coexpress_high").sum()

    assert n_coexpress == 258
    assert n_fru_only + n_coexpress == 4858
    assert n_dsx_only + n_coexpress == 412
    assert n_fru_high_only + n_coexpress_high == 2804
    assert n_dsx_high_only + n_coexpress_high == 331


def test_gate_verdict_is_alive_and_robust(real_traced_df):
    for rule in ("any_positive", "high_confidence_only"):
        summary = phase0_gate_summary(real_traced_df, rule)
        assert summary["verdict"] == "ALIVE"
        assert summary["n_residual_unexplained"] > 500
        assert "ol_intrinsic" in {r["category"] for r in summary["enriched_superclasses"]}
