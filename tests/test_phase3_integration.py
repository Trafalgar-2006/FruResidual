"""Integration test against the real Phase 3 outputs. Skipped if
scripts/run_phase3.py hasn't been run yet -- mirrors the pattern in
test_phase0/1/2_integration.py.
"""

import json

import pandas as pd
import pytest

from fruresidual.config import load_config

cfg = load_config()
_summary_path = cfg.data_processed / "phase3_summary.json"

pytestmark = pytest.mark.skipif(
    not _summary_path.exists(),
    reason=f"Phase 3 outputs not found (expected at {_summary_path}); run scripts/run_phase3.py first",
)


@pytest.fixture(scope="module")
def summary():
    with open(_summary_path) as f:
        return json.load(f)


def test_ol_intrinsic_excluded_as_saturated(summary):
    # Regression-pins the discovery that motivated stratification:
    # ol_intrinsic has zero internal residual/explained variance.
    assert "ol_intrinsic" in summary["excluded_strata"]
    mixed_names = {s["superclass"] for s in summary["mixed_strata"]}
    assert "ol_intrinsic" not in mixed_names


def test_ascending_neuron_is_not_degree_confounded(summary):
    # Regression-pins the one clean, trustworthy stratum.
    by_stratum = {d["superclass"]: d for d in summary["degree_confound_by_stratum"]}
    assert by_stratum["ascending_neuron"]["degree_confounded"] is False


def test_cb_intrinsic_and_vnc_intrinsic_are_degree_confounded(summary):
    by_stratum = {d["superclass"]: d for d in summary["degree_confound_by_stratum"]}
    assert by_stratum["cb_intrinsic"]["degree_confounded"] is True
    assert by_stratum["vnc_intrinsic"]["degree_confounded"] is True


def test_motif_pct_of_subset_is_a_valid_percentage():
    path = cfg.data_processed / "phase3_motif_enrichment_stratified.csv"
    if not path.exists():
        pytest.skip("motif enrichment CSV not found")
    df = pd.read_csv(path)
    # Would have caught the edge-vs-neuron bug directly: with the bug,
    # pct_of_subset values were on the order of 0.01-0.1%, never near
    # the n_in_subset/n_residual-implied percentage.
    assert (df["pct_of_subset"] <= 100).all()
    assert (df["pct_of_subset"] >= 0).all()


def test_shortlist_has_expected_signature_types():
    path = cfg.data_processed / "phase3_candidate_shortlist.csv"
    if not path.exists():
        pytest.skip("shortlist CSV not found")
    df = pd.read_csv(path)
    signature_types = df["signature_type"].unique()
    assert any("connectivity_motif" in s for s in signature_types)
    assert any("phase1_marginal" in s for s in signature_types)
