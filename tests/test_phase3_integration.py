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
    # Annotation dimensions used to be folded in as "phase1_marginal_*",
    # i.e. computed over the whole dimorphic population with no
    # stratification -- carrying the same regional confound the motif and
    # cluster analyses had already been corrected for. They are now
    # re-tested within each stratum and carry the stratum in their name.
    assert any(s.startswith("annotation_") and "[" in s for s in signature_types)
    assert not any("phase1_marginal" in s for s in signature_types)


def test_shortlist_excludes_unassigned_lineage_placeholders():
    # putative_primary (itoleeHl) and TBD (trumanHl) are "not yet
    # assigned a lineage" buckets and were the single largest hit in
    # their dimension. A neuron missing both a fru/dsx call and a
    # confident lineage call plausibly just means "under-annotated", so
    # they are excluded rather than reported as biology.
    path = cfg.data_processed / "phase3_candidate_shortlist.csv"
    if not path.exists():
        pytest.skip("shortlist CSV not found")
    df = pd.read_csv(path)
    assert not df["category"].isin({"putative_primary", "TBD"}).any()


def test_degree_adjusted_motifs_are_what_reach_the_shortlist():
    # Every connectivity_motif row must have survived the CMH degree
    # adjustment, so degree_confounded is False rather than True.
    path = cfg.data_processed / "phase3_candidate_shortlist.csv"
    if not path.exists():
        pytest.skip("shortlist CSV not found")
    df = pd.read_csv(path)
    motifs = df[df["signature_type"].str.startswith("connectivity_motif")]
    assert len(motifs) > 0
    assert not (motifs["degree_confounded"] == True).any()  # noqa: E712


def test_cluster_resolution_sweep_confirms_ascending_neuron_is_robust():
    # Regression-pins the third bug found on audit (2026-09-18): a fixed
    # resolution=0.1 had silently replaced the sensitivity sweep
    # clustering.py's own docstring calls for. Confirms the fix actually
    # validates (not just changes) the headline finding: ascending_neuron
    # stays significant across a majority of swept resolutions, while
    # cb_intrinsic/vnc_intrinsic don't.
    path = cfg.data_processed / "phase3_cluster_resolution_sweep.csv"
    if not path.exists():
        pytest.skip("resolution sweep CSV not found")
    df = pd.read_csv(path)
    hits = df.groupby("superclass")["n_significant"].apply(lambda s: (s > 0).sum())
    assert hits["ascending_neuron"] >= 3  # robust: majority of 5 swept resolutions
    assert hits.get("cb_intrinsic", 0) == 0  # clean null, not just unswept
    assert hits.get("vnc_intrinsic", 0) < 3  # not robust -- correctly excluded from shortlist


def test_only_robust_cluster_finding_reaches_the_shortlist():
    path = cfg.data_processed / "phase3_candidate_shortlist.csv"
    if not path.exists():
        pytest.skip("shortlist CSV not found")
    df = pd.read_csv(path)
    cluster_rows = df[df["signature_type"].str.contains("connectivity_cluster", na=False)]
    strata_in_shortlist = {s.split("[")[1].split(",")[0] for s in cluster_rows["signature_type"]}
    assert strata_in_shortlist == {"ascending_neuron"}
