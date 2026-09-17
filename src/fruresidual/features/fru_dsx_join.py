"""Phase-0 kill-gate logic: is the residual (sex-different neurons that
fru/dsx doesn't explain) big and structured, or noise?

Originally the plan (docs/project_plan.md section e, risk #2) expected a
hard problem here: dimorphism assumed to live at the cell-TYPE level while
fru/dsx is per-NEURON, requiring us to build our own type->neuron rollup.
In practice, MaleCNS's own bulk annotation file
(`body-annotations-male-cns-v1.0-minconf-0.5.feather`) already ships both
`dimorphism` and `fruDsx` per neuron (bodyId), pre-joined by the Janelia
team — confirmed by inspection on 2026-09-17. So there is no type-level
rollup to build; the "aggregation rule" that remains is just how to treat
`fruDsx`'s confidence suffix (`_high` vs `_low`).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fruresidual.config import Config
from fruresidual.eval.enrichment import category_enrichment

AggregationRule = Literal["high_confidence_only", "any_positive"]

# Raw fruDsx values observed in body-annotations-male-cns-v1.0-minconf-0.5.feather:
# {None, fru_high, fru_low, dsx_high, dsx_low, coexpress_high, coexpress_low}
_POSITIVE_VALUES: dict[AggregationRule, frozenset[str]] = {
    "any_positive": frozenset(
        {"fru_high", "fru_low", "dsx_high", "dsx_low", "coexpress_high", "coexpress_low"}
    ),
    "high_confidence_only": frozenset({"fru_high", "dsx_high", "coexpress_high"}),
}

# Non-null `dimorphism` values are the sex-different set (see body-annotations
# schema: "male-specific", "sexually dimorphic", "potentially male-specific",
# "potentially sexually dimorphic"; null == isomorphic).
DIMORPHIC_CATEGORIES = frozenset(
    {"male-specific", "sexually dimorphic", "potentially male-specific", "potentially sexually dimorphic"}
)


def load_male_cns_annotations(cfg: Config, traced_only: bool = True) -> pd.DataFrame:
    """Loads the bulk MaleCNS per-neuron annotation table.

    Assumes scripts/download_male_cns.py has already fetched
    body-annotations-male-cns-v1.0-minconf-0.5.feather into cfg.data_raw.
    """
    path = cfg.data_raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    df = pd.read_feather(path)
    if traced_only:
        df = df[df["status"] == "Traced"].copy()
    return df


def has_fru_dsx(df: pd.DataFrame, rule: AggregationRule) -> pd.Series:
    return df["fruDsx"].isin(_POSITIVE_VALUES[rule])


def build_dimorphism_crosstab(df: pd.DataFrame, rule: AggregationRule) -> pd.DataFrame:
    """The 2x2(+) [dimorphism category] x [carries fru/dsx yes/no] table that
    is the Phase-0 gate criterion, plus a pct_no_fru_dsx column.
    """
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    ct = pd.crosstab(dimorphic["dimorphism"], dimorphic["has_fru_dsx"])
    for col in (True, False):
        if col not in ct.columns:
            ct[col] = 0
    ct["pct_no_fru_dsx"] = (ct[False] / (ct[False] + ct[True]) * 100).round(1)
    return ct.sort_values("pct_no_fru_dsx", ascending=False)


def phase0_gate_summary(
    df: pd.DataFrame,
    rule: AggregationRule,
    alpha: float = 0.05,
    min_odds_ratio: float = 2.0,
    min_residual: int = 100,
) -> dict:
    """Computes the kill-gate verdict: is the unexplained residual both
    non-trivial in size AND structured?

    "Structured" is decided by a real statistical test, not an arbitrary
    threshold: Fisher's exact test per superclass, residual vs. the full
    dimorphic/male-specific population it was drawn from, Benjamini-Hochberg
    corrected across superclasses (fruresidual.eval.enrichment). At least
    one superclass must be significantly over-represented (p_adj < alpha)
    with a non-trivial effect size (odds ratio >= min_odds_ratio) — the
    same method the plan specifies for Phase 3's enrichment analysis,
    applied here at a lighter scope just to ground the gate decision.
    """
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    residual = dimorphic[~dimorphic["has_fru_dsx"]]

    n_dimorphic = len(dimorphic)
    n_residual = len(residual)

    enrichment = category_enrichment(
        residual["superclass"], dimorphic["superclass"], alpha=alpha, min_odds_ratio=min_odds_ratio
    )
    significant = enrichment[enrichment["significant"]]
    structured = len(significant) > 0

    return {
        "rule": rule,
        "n_dimorphic_or_male_specific": int(n_dimorphic),
        "n_residual_unexplained": int(n_residual),
        "pct_residual": round(100 * n_residual / n_dimorphic, 1) if n_dimorphic else 0.0,
        "enriched_superclasses": significant[
            ["category", "n_in_subset", "pct_of_subset", "odds_ratio", "p_adj"]
        ].to_dict("records"),
        "structured": bool(structured),
        "verdict": "ALIVE" if (n_residual >= min_residual and structured) else "INCONCLUSIVE",
    }
