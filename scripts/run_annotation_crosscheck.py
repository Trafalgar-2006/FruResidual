"""Biology vs. ascertainment bias: re-check the residual against
FlyWire's own fru/dsx annotation.

This addresses the project's central open question (see
docs/phase1_results.md "Interpretation -- and the standing ambiguity"):
is the fru/dsx-unexplained residual real biology, or an artifact of where
MaleCNS's annotation effort was spent? See
fruresidual.features.annotation_crosscheck for the logic and the two
confounds it controls.

Two annotation sources are supported:

  --source curated  (default) the `fru_dsx` column of the curated
      flywire_annotations table. Already run: it turns out to be NESTED
      inside MaleCNS's annotation (it never fires where MaleCNS didn't),
      so it cannot answer the question. Kept because demonstrating that
      is itself the result.

  --source codex    FlyWire Codex's `gene` field, exported via Codex ->
      Info -> Download Data and saved into data_raw. Different
      annotation lineage, so this one CAN answer the question -- if its
      independence_diagnostic comes back PARTIALLY INDEPENDENT.

Usage:  python scripts/run_annotation_crosscheck.py [--source curated|codex]
                                                    [--codex-file NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fruresidual.config import load_config
from fruresidual.data.flywire_client import (
    load_codex_annotations,
    load_codex_gene_matches,
    load_flywire_annotations,
)
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


_CODEX_HELP = """
Two ways to produce it, either works:

  (A) SEARCH EXPORT -- usually easiest
      codex.flywire.ai, make sure the dataset selector (top right) says
      FAFB v783, then search:   gene == Fruitless
      Download the result list, repeat for   gene == Doublesex   and
      concatenate, or search   gene == Fruitless || gene == Doublesex
      The file only needs a root_id (or cell_type) column -- it is the
      list of POSITIVES; the full population comes from the cached
      curated table.

  (B) FULL DUMP
      codex.flywire.ai -> Download Data -> the annotation/cell-info
      export that carries a `gene` column for every neuron.

Save either as {path} (.csv/.csv.gz/.tsv/.feather all fine). The loader
detects which of the two it got. Column names are auto-detected.

This is the one source that can settle the biology-vs-ascertainment
question -- see docs/annotation_crosscheck_results.md.
"""


def _load_source(cfg, source: str, codex_file: str) -> tuple[pd.DataFrame, str]:
    if source == "curated":
        return load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather"), "flywire_curated"

    path = cfg.data_raw / codex_file
    if not path.exists():
        raise SystemExit(f"Codex export not found at {path}.\n" + _CODEX_HELP.format(path=path))

    # A full dump carries a gene column for every neuron; a search export
    # is just the positives and has to be joined onto the population.
    # Try the full-dump reader first and fall back rather than making the
    # user declare which kind of file they downloaded.
    try:
        fw = load_codex_annotations(path)
        return fw, f"codex_full_dump[{codex_file}]"
    except ValueError:
        curated = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
        fw = load_codex_gene_matches(path, curated)
        label = (f"codex_search_export[{codex_file}, matched_on={fw.attrs['matched_on']}, "
                 f"{fw.attrs['n_matched']}/{fw.attrs['n_in_export']} matched]")
        return fw, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["curated", "codex"], default="curated")
    parser.add_argument("--codex-file", default="codex_annotations.csv")
    args = parser.parse_args()

    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    cfg.data_processed.mkdir(parents=True, exist_ok=True)

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    print(f"MaleCNS dimorphic: {len(dimorphic)}, "
          f"residual: {int((~dimorphic['has_fru_dsx']).sum())}, "
          f"explained: {int(dimorphic['has_fru_dsx'].sum())}")

    fw, source_label = _load_source(cfg, args.source, args.codex_file)
    print(f"\nannotation source: {source_label}  ({len(fw)} rows, "
          f"{int(fw['fru_dsx'].notna().sum())} fru/dsx-positive)")
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
        "annotation_source": source_label,
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
    # Suffixed per source so a codex run cannot silently overwrite the
    # curated run's output (or vice versa) -- they are different evidence.
    suffix = "" if args.source == "curated" else f"_{args.source}"
    out = cfg.data_processed / f"annotation_crosscheck_summary{suffix}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    strat.to_csv(cfg.data_processed / f"annotation_crosscheck_by_super_class{suffix}.csv", index=False)
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
