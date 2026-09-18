"""Biology vs. ascertainment bias: re-check the residual against
FlyWire's own fru/dsx annotation.

This addresses the project's central open question (see
docs/phase1_results.md "Interpretation -- and the standing ambiguity"):
is the fru/dsx-unexplained residual real biology, or an artifact of where
MaleCNS's annotation effort was spent? See
fruresidual.features.annotation_crosscheck for the logic and the two
confounds it controls.

Usage:  python scripts/run_annotation_crosscheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fruresidual.config import load_config
from fruresidual.data.flywire_client import load_flywire_annotations
from fruresidual.features.annotation_crosscheck import (
    build_flywire_type_lookup,
    crosscheck_residual,
    flywire_coverage_by_super_class,
    independence_diagnostic,
    match_dimorphic_to_flywire,
)
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)

# A stratum only tells us something if FlyWire actually annotates fru/dsx
# there; below this, "0% positive" is FlyWire's blind spot, not evidence.
MIN_FLYWIRE_COVERAGE_PCT = 1.0


def main() -> None:
    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    cfg.data_processed.mkdir(parents=True, exist_ok=True)

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    print(f"MaleCNS dimorphic: {len(dimorphic)}, "
          f"residual: {int((~dimorphic['has_fru_dsx']).sum())}, "
          f"explained: {int(dimorphic['has_fru_dsx'].sum())}")

    fw = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
    coverage = flywire_coverage_by_super_class(fw)
    print("\n=== FlyWire's OWN fru/dsx coverage by super_class ===")
    print("(a stratum with near-zero coverage cannot inform this test)")
    print(coverage[["n_fru_dsx_positive", "n_neurons", "pct_positive"]].to_string())

    lookup = build_flywire_type_lookup(fw)
    matched = match_dimorphic_to_flywire(dimorphic, lookup)
    print(f"\nmatched to a real FlyWire cell_type: {len(matched)}/{len(dimorphic)}")

    indep = independence_diagnostic(matched)
    print("\n=== Independence diagnostic ===")
    print("flywire_annotations' fru_dsx was built with reference to MaleCNS, so a")
    print("wholesale copy would never disagree. How often do they actually differ?")
    for k, v in indep.items():
        print(f"  {k}: {v}")

    result = crosscheck_residual(matched)
    ov = result["overall"]
    print("\n=== THE TEST: does FlyWire independently call the residual fru/dsx+? ===")
    print(f"  residual : {ov['n_residual_flywire_positive']}/{ov['n_residual']} "
          f"FlyWire-positive ({ov['pct_residual_flywire_positive']}%)")
    print(f"  explained: {ov['n_explained_flywire_positive']}/{ov['n_explained']} "
          f"FlyWire-positive ({ov['pct_explained_flywire_positive']}%)")
    print(f"  expected residual positives if it behaved like explained: "
          f"{ov['expected_residual_positive_at_explained_rate']}")
    print(f"  Fisher (residual depleted): p={ov['p_residual_depleted']:.3e}")

    print("\n=== Stratified by FlyWire super_class (controls shared coverage) ===")
    rows = []
    for sc, r in result["by_flywire_super_class"].items():
        cov_pct = float(coverage["pct_positive"].get(sc, 0.0))
        rows.append({
            "flywire_super_class": sc, "flywire_coverage_pct": cov_pct,
            "n_residual": r["n_residual"], "pct_residual_fw_pos": r["pct_residual_flywire_positive"],
            "n_explained": r["n_explained"], "pct_explained_fw_pos": r["pct_explained_flywire_positive"],
            "p_residual_depleted": r["p_residual_depleted"],
            "informative": cov_pct >= MIN_FLYWIRE_COVERAGE_PCT,
        })
    strat = pd.DataFrame(rows).sort_values("flywire_coverage_pct", ascending=False)
    print(strat.to_string(index=False))

    # Restricted to strata where FlyWire genuinely annotates fru/dsx
    informative_scs = set(coverage[coverage["pct_positive"] >= MIN_FLYWIRE_COVERAGE_PCT].index)
    sub = matched[matched["fw_super_class"].isin(informative_scs)]
    restricted = crosscheck_residual(sub)["overall"]
    print(f"\n=== Restricted to strata with >={MIN_FLYWIRE_COVERAGE_PCT}% FlyWire coverage ===")
    print(f"  (the informative subset: {sorted(informative_scs)})")
    print(f"  residual : {restricted['n_residual_flywire_positive']}/{restricted['n_residual']} "
          f"({restricted['pct_residual_flywire_positive']}%)")
    print(f"  explained: {restricted['n_explained_flywire_positive']}/{restricted['n_explained']} "
          f"({restricted['pct_explained_flywire_positive']}%)")
    print(f"  expected residual positives at the explained rate: "
          f"{restricted['expected_residual_positive_at_explained_rate']}")
    print(f"  Fisher: p={restricted['p_residual_depleted']:.3e}")

    # The p-value is NOT the deciding factor. If FlyWire's labels are
    # nested inside MaleCNS's, a vanishing p-value is guaranteed by
    # construction and means nothing about biology.
    if indep["flywire_labels_nested_within_malecns"]:
        verdict = (
            "INCONCLUSIVE -- SECOND ANNOTATION IS NOT INDEPENDENT. FlyWire's fru_dsx "
            f"labels are a strict subset of MaleCNS's ({indep['n_malecns_neg_flywire_pos']} "
            "cases of FlyWire-positive where MaleCNS was negative). P(FlyWire+ | MaleCNS-) "
            "is therefore 0 by construction, so the residual's 0% cannot distinguish "
            "biology from ascertainment bias, however small the p-value. Resolving this "
            "needs a genuinely independent fru/dsx source -- see docs/annotation_crosscheck_results.md."
        )
    elif restricted["p_residual_depleted"] < 0.05:
        verdict = (
            "BIOLOGY-LEANING: a partially independent annotation agrees the residual lacks "
            "fru/dsx, in strata where FlyWire demonstrably does annotate it"
        )
    else:
        verdict = "INCONCLUSIVE: residual not significantly depleted vs explained"
    print(f"\n  VERDICT: {verdict}")

    summary = {
        "rule": rule,
        "n_dimorphic": len(dimorphic),
        "n_matched_to_flywire": len(matched),
        "min_flywire_coverage_pct": MIN_FLYWIRE_COVERAGE_PCT,
        "flywire_coverage_by_super_class": coverage["pct_positive"].to_dict(),
        "independence_diagnostic": indep,
        "overall": ov,
        "restricted_to_informative_strata": restricted,
        "by_flywire_super_class": result["by_flywire_super_class"],
        "verdict": verdict,
    }
    out = cfg.data_processed / "annotation_crosscheck_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    strat.to_csv(cfg.data_processed / "annotation_crosscheck_by_super_class.csv", index=False)
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
