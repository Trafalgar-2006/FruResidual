"""Integration test against the real Phase 4 outputs. Skipped if
scripts/run_phase4.py hasn't been run yet -- mirrors the pattern in
test_phase0/1/2/3_integration.py.
"""

import json

import pytest

from fruresidual.config import load_config

cfg = load_config()
_summary_path = cfg.data_processed / "phase4_summary.json"

pytestmark = pytest.mark.skipif(
    not _summary_path.exists(),
    reason=f"Phase 4 outputs not found (expected at {_summary_path}); run scripts/run_phase4.py first",
)


@pytest.fixture(scope="module")
def summary():
    with open(_summary_path) as f:
        return json.load(f)


def test_matched_population_is_a_meaningful_size(summary):
    # Regression-pins the real matching yield: ~14K unambiguous 1:1
    # BANC<->MaleCNS matches were found within the 165K traced population.
    assert summary["n_unambiguous_banc_matches"] > 10_000


def test_recall_is_high_precision_is_not(summary):
    # Regression-pins the real finding: BANC confirms most MaleCNS
    # dimorphism calls (high recall) but calls far more neurons dimorphic
    # overall than MaleCNS does within the matched set (low precision) --
    # a real asymmetry between the two groups' classification criteria,
    # not a bug. Conflating the two (e.g. reporting only Jaccard) would
    # misread "BANC found more candidates" as "BANC disagrees with us."
    overall = summary["overall"]
    assert overall["recall_malecns_dimorphic_confirmed_by_banc"] > 0.7
    assert overall["precision_banc_dimorphic_matches_malecns"] < overall["recall_malecns_dimorphic_confirmed_by_banc"]


def test_high_recall_is_not_an_artifact_of_missing_banc_calls(summary):
    # BANC leaves 70K+ neurons unassessed (null sexually_dimorphic), which
    # the concordance math counts as "not confirmed" exactly like a real
    # disagreement. If the matched dimorphic set were mostly unassessed,
    # the headline recall would be measuring missing data, not agreement.
    # It isn't: essentially all matches carry a real BANC call, so
    # restricting to assessed-only barely moves recall.
    overall = summary["overall"]
    n_md = overall["n_malecns_dimorphic_matched"]
    assert overall["n_malecns_dimorphic_unassessed_by_banc"] < 0.05 * n_md
    assert abs(
        overall["recall_among_banc_assessed_only"]
        - overall["recall_malecns_dimorphic_confirmed_by_banc"]
    ) < 0.05
    # and the finding doesn't rest on unproofread reconstructions
    assert overall["n_malecns_dimorphic_unproofread_in_banc"] == 0


def test_residual_not_significantly_different_from_explained(summary):
    # The actual Phase 4 finding: residual (fru/dsx-unexplained) neurons
    # are confirmed dimorphic by BANC at a statistically indistinguishable
    # rate from explained neurons -- i.e. the residual isn't disproportionately
    # made of false positives relative to the rest of the dimorphic population.
    overall = summary["overall"]
    assert overall["residual_vs_explained_p_value"] > 0.05
    assert overall["pct_residual_confirmed_dimorphic_by_banc"] > 70
    assert overall["pct_explained_confirmed_dimorphic_by_banc"] > 70
