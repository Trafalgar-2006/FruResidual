"""Integration test against the real Phase 1 outputs (cached ROI data +
MaleCNS annotations). Skipped if scripts/run_phase1.py hasn't been run yet
-- mirrors the pattern in test_phase0_integration.py.
"""

import pandas as pd
import pytest

from fruresidual.config import load_config
from fruresidual.features.atlas import parse_roi_info, roi_residual_summary
from fruresidual.features.fru_dsx_join import DIMORPHIC_CATEGORIES, has_fru_dsx, load_male_cns_annotations

cfg = load_config()
_roi_cache = cfg.data_interim / "phase1_roi_info.feather"

pytestmark = pytest.mark.skipif(
    not _roi_cache.exists(),
    reason=f"Phase 1 ROI cache not found (expected at {_roi_cache}); run scripts/run_phase1.py first",
)


@pytest.fixture(scope="module")
def roi_summary():
    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, "any_positive")
    residual_ids = set(dimorphic.loc[~dimorphic["has_fru_dsx"], "bodyId"])

    roi_raw = pd.read_feather(_roi_cache)
    roi_long = parse_roi_info(roi_raw["roiInfo"], roi_raw["bodyId"])
    return roi_residual_summary(roi_long, residual_ids)


def test_no_fine_grained_sub_rois_leak_through(roi_summary):
    assert not roi_summary.index.str.contains("_col_|_layer_|-", regex=True).any()


def test_optic_lobe_neuropils_are_almost_entirely_residual(roi_summary):
    # Regression-pins the headline Phase 1 finding: ME/LO/Optic neuropils
    # are ~95-100% unexplained by fru/dsx.
    for roi in ("ME(R)", "ME(L)", "LO(R)", "LO(L)", "Optic(R)", "Optic(L)"):
        assert roi_summary.loc[roi, "pct_residual"] > 90


def test_central_brain_integrative_neuropils_are_mostly_explained(roi_summary):
    # SIP/AVLP are the low end of the spectrum -- mostly explained by fru/dsx.
    for roi in ("SIP(R)", "SIP(L)", "AVLP(R)", "AVLP(L)"):
        assert roi_summary.loc[roi, "pct_residual"] < 30
