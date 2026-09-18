"""Integration test against the real Phase 2 outputs. Skipped if
scripts/run_phase2_data.py and scripts/run_phase2.py haven't been run yet
-- mirrors the pattern in test_phase0/1_integration.py.
"""

import pandas as pd
import pytest

from fruresidual.config import load_config

cfg = load_config()
_targets_path = cfg.data_processed / "phase2_wiring_targets.feather"
_residuals_path = cfg.data_processed / "phase2_residual_scores.feather"

pytestmark = pytest.mark.skipif(
    not _targets_path.exists(),
    reason=f"Phase 2 targets not found (expected at {_targets_path}); "
    "run scripts/run_phase2_data.py first",
)


@pytest.fixture(scope="module")
def targets():
    return pd.read_feather(_targets_path)


def test_wiring_difference_in_valid_range(targets):
    assert (targets["wiring_difference"] >= 0).all()
    assert (targets["wiring_difference"] <= 1).all()


def test_no_duplicate_neurons(targets):
    assert targets["bodyId"].duplicated().sum() == 0


def test_case_control_both_present(targets):
    assert targets["is_case"].sum() > 0
    assert (~targets["is_case"]).sum() > 0


def test_cases_show_higher_wiring_difference_than_controls(targets):
    # Sanity check on the target's own construction: dimorphic (case)
    # neurons should, on average, look MORE different from their FlyWire
    # counterpart than isomorphic (control) neurons do -- if this fails,
    # the wiring-difference metric itself is broken, independent of any
    # model's ability to predict it.
    case_mean = targets.loc[targets["is_case"], "wiring_difference"].mean()
    control_mean = targets.loc[~targets["is_case"], "wiring_difference"].mean()
    assert case_mean > control_mean


@pytest.mark.skipif(not _residuals_path.exists(), reason="run scripts/run_phase2.py first")
def test_residual_scores_cover_the_population():
    residuals = pd.read_feather(_residuals_path)
    targets_df = pd.read_feather(_targets_path)
    assert len(residuals) == len(targets_df)
    assert residuals["bodyId"].isin(targets_df["bodyId"]).all()
