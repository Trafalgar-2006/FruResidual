"""Phase 1: where does the fru/dsx-unexplained residual concentrate?

Two complementary views, both built on the same residual/dimorphic split
from Phase 0 (fruresidual.features.fru_dsx_join):
  - ROI/neuropil level (fine-grained, spatial) -- fetched live from
    neuPrint's roiInfo, targeted to just the dimorphic/male-specific
    neurons (~2,368), not the full 165K-neuron connectome.
  - Categorical dimensions already in the bulk annotation file (superclass,
    class, hemilineage, somaNeuromere, entryNerve/exitNerve, side) plus
    neurotransmitter (joined from body-neurotransmitters-male-cns-v1.0),
    tested via the same fruresidual.eval.enrichment machinery Phase 0 used.
"""

from __future__ import annotations

import json

import pandas as pd

from fruresidual.config import Config
from fruresidual.eval.enrichment import category_enrichment


def parse_roi_info(roi_info_series: pd.Series, bodyid_series: pd.Series) -> pd.DataFrame:
    """Expands neuPrint's per-neuron roiInfo JSON blobs into long format:
    one row per (bodyId, roi, synweight). synweight (pre+post synapse
    count) is used as the innervation-strength measure.
    """
    rows = []
    for body_id, raw in zip(bodyid_series, roi_info_series):
        if not raw:
            continue
        roi_dict = json.loads(raw) if isinstance(raw, str) else raw
        for roi, counts in roi_dict.items():
            rows.append({"bodyId": body_id, "roi": roi, "synweight": counts.get("synweight", 0)})
    return pd.DataFrame(rows)


def roi_residual_summary(roi_long: pd.DataFrame, residual_body_ids: set[int]) -> pd.DataFrame:
    """Per-ROI breakdown of synapse weight contributed by residual
    (unexplained) vs. explained dimorphic/male-specific neurons.

    Excludes fine sub-neuropil parcellations: optic-lobe columns
    ("ME_R_col_30_31"), optic-lobe layers ("ME_L_layer_08"), and
    antennal-lobe glomeruli ("AL-VA1d(R)") -- together ~4,386 of the
    ~4,560 raw ROI entries in this dataset. They trivially inherit their
    parent neuropil's residual rate at far higher cardinality and add no
    information over the coarse neuropil names (e.g. "ME(R)", "AL(R)")
    they subdivide.

    CAUTION: neuPrint's remaining ROI names are still a nested hierarchy,
    not disjoint regions -- e.g. a synapse in "SIP(R)" is also counted
    under the parent aggregate "CentralBrain". Verified directly: summing
    named central-brain sub-regions gives ~1.3x "CentralBrain"'s own
    total. Each row's pct_residual is still a valid ratio computed within
    that one ROI; don't sum/compare raw synweight totals ACROSS rows as
    if they were disjoint parts of a whole, since they aren't.

    `synweight` is neuPrint's own weighted composite of
    pre/post/downstream/upstream synapse counts, not a raw synapse
    tally -- roughly 2-3x the simple pre+post sum in spot checks.
    """
    is_fine_grained = roi_long["roi"].str.contains("_col_|_layer_|-", regex=True)
    roi_long = roi_long[~is_fine_grained].copy()
    roi_long["is_residual"] = roi_long["bodyId"].isin(residual_body_ids)
    summary = (
        roi_long.groupby(["roi", "is_residual"])["synweight"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={True: "residual_synweight", False: "explained_synweight"})
    )
    for col in ("residual_synweight", "explained_synweight"):
        if col not in summary.columns:
            summary[col] = 0
    summary["total_synweight"] = summary["residual_synweight"] + summary["explained_synweight"]
    summary["pct_residual"] = (
        100 * summary["residual_synweight"] / summary["total_synweight"]
    ).round(1)
    return summary.sort_values("total_synweight", ascending=False)


def load_neurotransmitters(cfg: Config) -> pd.DataFrame:
    """Loads body-neurotransmitters-male-cns-v1.0.feather and returns just
    (body, consensus_nt) -- the file covers all raw body fragments (1.8M+
    rows), far more than the proofread neuron set, so callers should join
    on body/bodyId rather than assume row-alignment with other tables.
    """
    path = cfg.data_raw / "body-neurotransmitters-male-cns-v1.0.feather"
    df = pd.read_feather(path, columns=["body", "consensus_nt"])
    return df.rename(columns={"body": "bodyId"})


def multi_dimension_enrichment(
    dimorphic: pd.DataFrame, residual_mask: pd.Series, dimensions: list[str]
) -> dict[str, pd.DataFrame]:
    """Runs fruresidual.eval.enrichment.category_enrichment separately for
    each column in `dimensions` (each dimension is its own BH-correction
    family, standard practice for enrichment sweeps across unrelated
    annotation axes).
    """
    residual = dimorphic[residual_mask]
    results = {}
    for dim in dimensions:
        sub = residual[dim].dropna()
        bg = dimorphic[dim].dropna()
        if sub.empty or bg.empty:
            continue
        results[dim] = category_enrichment(sub, bg)
    return results
