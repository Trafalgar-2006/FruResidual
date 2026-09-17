import json

import pandas as pd
import pytest

from fruresidual.features.atlas import multi_dimension_enrichment, parse_roi_info, roi_residual_summary


def test_parse_roi_info_expands_json_to_long_format():
    roi_info = pd.Series(
        [
            json.dumps({"ME(R)": {"synweight": 10}, "LO(R)": {"synweight": 5}}),
            json.dumps({"ME(R)": {"synweight": 3}}),
            None,
        ]
    )
    body_ids = pd.Series([1, 2, 3])
    long_df = parse_roi_info(roi_info, body_ids)
    assert len(long_df) == 3  # None row contributes nothing
    assert set(long_df["roi"]) == {"ME(R)", "LO(R)"}
    assert long_df.loc[long_df["bodyId"] == 1, "synweight"].sum() == 15


def test_roi_residual_summary_excludes_fine_grained_sub_rois():
    long_df = pd.DataFrame(
        {
            "bodyId": [1, 1, 2, 2],
            "roi": ["ME(R)", "ME_R_col_01_02", "ME(R)", "ME_R_layer_03"],
            "synweight": [100, 50, 200, 80],
        }
    )
    summary = roi_residual_summary(long_df, residual_body_ids={1})
    assert list(summary.index) == ["ME(R)"]  # sub-ROIs dropped
    assert summary.loc["ME(R)", "residual_synweight"] == 100
    assert summary.loc["ME(R)", "explained_synweight"] == 200
    assert summary.loc["ME(R)", "pct_residual"] == pytest.approx(100 / 300 * 100, abs=0.1)


def test_multi_dimension_enrichment_skips_empty_dimensions():
    df = pd.DataFrame(
        {
            "superclass": ["a"] * 50 + ["b"] * 50,
            "empty_dim": [None] * 100,
        }
    )
    residual_mask = pd.Series([True] * 20 + [False] * 30 + [True] * 10 + [False] * 40)
    results = multi_dimension_enrichment(df, residual_mask, ["superclass", "empty_dim"])
    assert "superclass" in results
    assert "empty_dim" not in results  # all-null column produces no result
