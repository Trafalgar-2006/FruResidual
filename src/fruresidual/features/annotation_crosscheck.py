"""Does a SECOND fru/dsx annotation, made by a different group on a
different animal, also call the residual neurons fru/dsx-negative?

This is the project's central open question (docs/phase1_results.md):
the residual could be real biology, or it could be an artifact of where
MaleCNS's fru/dsx annotation effort was spent (~75% in the central
brain). A single annotation source cannot distinguish them. FlyWire's
`flywire_annotations` table carries its own `fru_dsx` column, so the
residual can be re-checked against it.

The logic:
  - residual neurons' FlyWire counterparts often fru/dsx-POSITIVE
    => MaleCNS missed them => ASCERTAINMENT BIAS
  - residual negative in BOTH annotations
    => two annotations agree => evidence for BIOLOGY

Two things have to be controlled for this to mean anything, and both are
implemented here rather than assumed:

1. **Shared coverage.** If FlyWire also barely annotates fru/dsx where
   the residual lives, "negative in both" is just the same blind spot
   twice. Handled by stratifying on FlyWire's own `super_class` and by
   reporting FlyWire's per-super_class coverage alongside the result, so
   strata where FlyWire genuinely does annotate can be read separately.

2. **Annotation independence.** docs/project_plan.md section (b.2) notes
   the `fru_dsx` column was added to flywire_annotations "via
   cross-validation against MaleCNS", so the two sources are NOT
   guaranteed independent, and a derived column would agree with MaleCNS
   by construction. `independence_diagnostic` measures how often the two
   actually DISAGREE: a pure copy would disagree zero times, so a
   substantial disagreement rate is evidence the FlyWire column carries
   information of its own. This bounds the concern; it does not remove
   it, and the provenance question is flagged in the writeup rather than
   resolved here.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact


def build_flywire_type_lookup(flywire_annotations: pd.DataFrame) -> pd.DataFrame:
    """Per FlyWire cell_type: does ANY neuron of that type carry a
    fru/dsx call, and which super_class does the type belong to.

    "Any" rather than "majority" on purpose -- it is the rule most likely
    to FIND a fru/dsx call on the residual, so it biases against this
    analysis's own conclusion rather than toward it.
    """
    fw = flywire_annotations.copy()
    fw["fw_fru_dsx_positive"] = fw["fru_dsx"].notna()
    return fw.groupby("cell_type").agg(
        fw_fru_dsx_any=("fw_fru_dsx_positive", "any"),
        fw_n_positive=("fw_fru_dsx_positive", "sum"),
        fw_n_neurons=("fw_fru_dsx_positive", "size"),
        fw_super_class=("super_class", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    )


def flywire_coverage_by_super_class(flywire_annotations: pd.DataFrame) -> pd.DataFrame:
    """FlyWire's own fru/dsx annotation density per super_class -- the
    denominator that decides whether a 0% result in a given stratum is
    informative or just FlyWire's blind spot."""
    fw = flywire_annotations.copy()
    fw["pos"] = fw["fru_dsx"].notna()
    out = fw.groupby("super_class")["pos"].agg(["sum", "size", "mean"])
    out.columns = ["n_fru_dsx_positive", "n_neurons", "fraction_positive"]
    out["pct_positive"] = (100 * out["fraction_positive"]).round(2)
    return out.sort_values("pct_positive", ascending=False)


def match_dimorphic_to_flywire(
    dimorphic: pd.DataFrame, type_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Restricts the MaleCNS dimorphic population to neurons with a
    single clean flywireType that exists in FlyWire's own vocabulary,
    and attaches FlyWire's independent verdict.

    Comma-containing flywireType values are ambiguous multi-type matches
    and are excluded rather than arbitrarily split -- the same rule
    fruresidual.features.rewiring uses.
    """
    clean = (
        dimorphic["flywireType"].notna()
        & ~dimorphic["flywireType"].str.contains(",", na=False)
        & dimorphic["flywireType"].isin(type_lookup.index)
    )
    m = dimorphic[clean].copy()
    m["fw_fru_dsx_any"] = m["flywireType"].map(type_lookup["fw_fru_dsx_any"])
    m["fw_super_class"] = m["flywireType"].map(type_lookup["fw_super_class"])
    return m


def independence_diagnostic(matched: pd.DataFrame) -> dict:
    """How often do the two annotations disagree?

    flywire_annotations' fru_dsx column was built with reference to
    MaleCNS, so it is not guaranteed independent. A column copied
    wholesale would agree with MaleCNS every time. Measuring the actual
    disagreement rate bounds how derivative it can be: substantial
    disagreement means the FlyWire column carries information MaleCNS
    does not. This does not prove independence, it only rules out the
    worst case.
    """
    mc_pos = matched["has_fru_dsx"].astype(bool)
    fw_pos = matched["fw_fru_dsx_any"].astype(bool)
    n = len(matched)
    mc_neg_fw_pos = int((~mc_pos & fw_pos).sum())
    mc_pos_fw_neg = int((mc_pos & ~fw_pos).sum())

    # The decisive diagnostic is the ASYMMETRY, not the overall rate.
    # If FlyWire never calls fru/dsx where MaleCNS didn't, its labels are
    # a subset of MaleCNS's -- nested, not independent. In that case
    # P(FlyWire+ | MaleCNS-) = 0 holds BY CONSTRUCTION, so a finding that
    # the residual is FlyWire-negative carries no information about
    # biology at all, however extreme its p-value looks.
    nested = mc_neg_fw_pos == 0 and mc_pos_fw_neg > 0
    return {
        "n_matched": n,
        "n_agree_positive": int((mc_pos & fw_pos).sum()),
        "n_agree_negative": int((~mc_pos & ~fw_pos).sum()),
        "n_malecns_pos_flywire_neg": mc_pos_fw_neg,
        "n_malecns_neg_flywire_pos": mc_neg_fw_pos,
        "disagreement_rate": float((mc_pos != fw_pos).mean()) if n else 0.0,
        "disagreement_rate_among_malecns_positive": (
            float((~fw_pos[mc_pos]).mean()) if int(mc_pos.sum()) else None
        ),
        "flywire_labels_nested_within_malecns": bool(nested),
        "independence_verdict": (
            "NOT INDEPENDENT (nested): FlyWire never calls fru/dsx where MaleCNS did not, "
            "so the residual being FlyWire-negative is partly true by construction and "
            "cannot adjudicate biology vs ascertainment"
            if nested else
            "PARTIALLY INDEPENDENT: FlyWire calls fru/dsx on neurons MaleCNS did not, "
            "so its column carries information of its own"
        ),
    }


def crosscheck_residual(matched: pd.DataFrame, stratum_col: str = "fw_super_class") -> dict:
    """The test itself, overall and stratified by FlyWire super_class.

    `matched` needs a boolean `has_fru_dsx` (MaleCNS side) and
    `fw_fru_dsx_any` (FlyWire side). Fisher is one-sided on "residual is
    DEPLETED of FlyWire fru/dsx calls relative to explained", which is
    the biology hypothesis; the ascertainment hypothesis predicts no
    depletion.
    """
    def _one(g: pd.DataFrame) -> dict:
        r = g[~g["has_fru_dsx"].astype(bool)]
        e = g[g["has_fru_dsx"].astype(bool)]
        rp, ep = int(r["fw_fru_dsx_any"].sum()), int(e["fw_fru_dsx_any"].sum())
        res = {
            "n_residual": len(r), "n_residual_flywire_positive": rp,
            "pct_residual_flywire_positive": round(100 * rp / len(r), 1) if len(r) else None,
            "n_explained": len(e), "n_explained_flywire_positive": ep,
            "pct_explained_flywire_positive": round(100 * ep / len(e), 1) if len(e) else None,
        }
        if len(r) and len(e):
            _, p = fisher_exact([[rp, len(r) - rp], [ep, len(e) - ep]], alternative="less")
            res["p_residual_depleted"] = float(p)
            # what we would see if the residual behaved like the explained set
            res["expected_residual_positive_at_explained_rate"] = round(len(r) * ep / len(e), 1)
        else:
            res["p_residual_depleted"] = float("nan")
            res["expected_residual_positive_at_explained_rate"] = None
        return res

    out = {"overall": _one(matched), "by_flywire_super_class": {}}
    for sc, g in matched.groupby(stratum_col):
        out["by_flywire_super_class"][str(sc)] = _one(g)
    return out
