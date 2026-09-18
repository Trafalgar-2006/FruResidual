"""Phase 4: does the residual finding survive an independent female
reconstruction? See docs/project_plan.md section (d), PHASE 4.

Scope note (see fruresidual.data.banc_client for the full story): BANC's
live connectivity API needs a per-dataset "view" permission this
project's account doesn't have, so the originally-planned approach
(re-derive the Phase 2 wiring-difference metric using BANC connectivity)
isn't available. Instead, this validates the more fundamental claim the
whole project rests on: BANC's bulk metadata table ships its own
independent `sexually_dimorphic` call plus direct neuron-level
`malecns_match` cross-references, so we can ask "does an independent
research group's independent female reconstruction agree that these
MaleCNS neurons are sexually dimorphic?" -- a real cross-dataset
validation, just of the dimorphism classification rather than the
wiring-difference score specifically.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact

# BANC's own independent classification -- concordance is checked
# against ANY of these (not isomorphic, not unclassified).
BANC_DIMORPHIC_VALUES = frozenset({"dimorphic", "female-specific", "male-specific"})


def match_banc_to_malecns(banc_df: pd.DataFrame, malecns_body_ids: set[int]) -> pd.DataFrame:
    """Restricts BANC to neurons with an unambiguous (exactly one BANC
    row) match to a MaleCNS bodyId in `malecns_body_ids`. ~80.6% of all
    malecns_match values are already 1:1 (verified 2026-09-19); the
    remaining ~19% (one MaleCNS neuron claimed by multiple BANC
    fragments) are excluded rather than arbitrarily resolved, since
    there's no principled way to pick among ambiguous fragment matches
    from this table alone.
    """
    matched = banc_df[banc_df["malecns_match"].notna()].copy()
    matched["bodyId"] = pd.to_numeric(matched["malecns_match"], errors="coerce").astype("Int64")
    matched = matched[matched["bodyId"].isin(malecns_body_ids)]

    unambiguous_ids = matched["bodyId"].value_counts()
    unambiguous_ids = unambiguous_ids[unambiguous_ids == 1].index
    matched = matched[matched["bodyId"].isin(unambiguous_ids)]

    return matched[["bodyId", "root_id", "cell_type", "sexually_dimorphic", "proofread"]].reset_index(drop=True)


def compute_concordance(
    matched: pd.DataFrame, malecns_dimorphism: pd.Series, residual_body_ids: set[int]
) -> dict:
    """malecns_dimorphism: Series indexed by bodyId, values are the raw
    `dimorphism` column (non-null = MaleCNS-side dimorphic/male-specific,
    null = isomorphic) for the SAME population `matched` was built from.

    Returns overlap/concordance statistics plus a Fisher's exact test on
    whether residual (fru/dsx-unexplained) neurons are more or less
    likely to be independently confirmed dimorphic by BANC than
    explained neurons are -- the actual Phase-4 question.
    """
    df = matched.copy()
    df["malecns_dimorphic"] = df["bodyId"].map(lambda b: pd.notna(malecns_dimorphism.get(b)))
    df["banc_dimorphic"] = df["sexually_dimorphic"].isin(BANC_DIMORPHIC_VALUES)
    df["is_residual"] = df["bodyId"].isin(residual_body_ids)

    both = int((df["malecns_dimorphic"] & df["banc_dimorphic"]).sum())
    either = int((df["malecns_dimorphic"] | df["banc_dimorphic"]).sum())
    jaccard = both / either if either else 0.0
    n_malecns_dimorphic = int(df["malecns_dimorphic"].sum())
    n_banc_dimorphic = int(df["banc_dimorphic"].sum())
    # Jaccard alone is misleading for a rare trait (dimorphism is ~1% of
    # neurons here): even near-perfect agreement gives a low Jaccard once
    # each side also independently flags trait-positive neurons the
    # other side doesn't share a match for. Recall/precision separate
    # "does BANC confirm what MaleCNS called dimorphic" (the actual
    # Phase-4 question) from "how much of BANC's own independent
    # dimorphic set this happens to include" (not the same question).
    recall = both / n_malecns_dimorphic if n_malecns_dimorphic else None
    precision = both / n_banc_dimorphic if n_banc_dimorphic else None

    residual_rows = df[df["is_residual"]]
    explained_rows = df[df["malecns_dimorphic"] & ~df["is_residual"]]

    a = int(residual_rows["banc_dimorphic"].sum())
    b = len(residual_rows) - a
    c = int(explained_rows["banc_dimorphic"].sum())
    d = len(explained_rows) - c
    if min(len(residual_rows), len(explained_rows)) > 0:
        _, p_value = fisher_exact([[a, b], [c, d]])
    else:
        p_value = float("nan")

    return {
        "n_matched_total": len(df),
        "n_malecns_dimorphic_matched": n_malecns_dimorphic,
        "n_banc_dimorphic_matched": n_banc_dimorphic,
        "jaccard_dimorphic_overlap": jaccard,
        "recall_malecns_dimorphic_confirmed_by_banc": recall,
        "precision_banc_dimorphic_matches_malecns": precision,
        "n_residual_matched": len(residual_rows),
        "pct_residual_confirmed_dimorphic_by_banc": round(100 * a / len(residual_rows), 1) if len(residual_rows) else None,
        "n_explained_matched": len(explained_rows),
        "pct_explained_confirmed_dimorphic_by_banc": round(100 * c / len(explained_rows), 1) if len(explained_rows) else None,
        "residual_vs_explained_p_value": p_value,
    }
