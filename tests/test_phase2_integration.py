"""Integration test against the real Phase 2 outputs. Skipped if
scripts/run_phase2_data.py and scripts/run_phase2.py haven't been run yet
-- mirrors the pattern in test_phase0/1_integration.py.
"""

import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

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
    # neurons should look MORE different from their FlyWire counterpart
    # than isomorphic (control) neurons do. If this fails, the metric
    # itself is broken, independent of any model's ability to predict it.
    #
    # Tested with Mann-Whitney rather than a bare mean comparison. The
    # distribution is heavily right-skewed and the controls carry a fatter
    # tail, so the two means sit close together (ratio ~1.05) even though
    # the rank separation is large and highly significant -- a mean-only
    # assertion would be one outlier away from flipping and would tell us
    # nothing about whether the signal is real.
    case = targets.loc[targets["is_case"], "wiring_difference"]
    control = targets.loc[~targets["is_case"], "wiring_difference"]
    _, p = mannwhitneyu(case, control, alternative="greater")
    assert p < 0.001
    assert case.median() > control.median()


def test_target_has_no_unscorable_placeholder_values(targets):
    # cosine_distance returns NaN for an empty profile and run_phase2_data
    # drops those rows, so nothing unscorable should reach the target. It
    # previously returned 1.0 instead, which would have silently planted
    # maximum-difference rows at the top of the distribution.
    assert not targets["wiring_difference"].isna().any()
    assert (targets["wiring_difference"] < 1.0).all()


@pytest.mark.skipif(not _residuals_path.exists(), reason="run scripts/run_phase2.py first")
def test_residual_scores_cover_the_population():
    residuals = pd.read_feather(_residuals_path)
    targets_df = pd.read_feather(_targets_path)
    assert len(residuals) == len(targets_df)
    assert residuals["bodyId"].isin(targets_df["bodyId"]).all()
