"""Phase 3: turn the residual into a hypothesis. See docs/project_plan.md
section (d), PHASE 3.

Stratified by superclass, and this is not optional: an earlier
unstratified version of this script found that ol_intrinsic is 100%
residual (0 explained neurons in that superclass) -- zero within-stratum
variance -- which makes any marginal (unstratified) comparison spuriously
flag every optic-lobe-specific partner type as "residual-enriched"
tautologically, not from any real circuit signal (see
test_regional_confound_produces_spurious_marginal_enrichment). Analysis
here only runs within superclasses that have >= MIN_GROUP_SIZE of BOTH
residual and explained neurons, where "why do these two neurons in the
same region differ" is a real, non-tautological question.

Requires the cached profiles from scripts/run_phase3_data.py (run that
first) -- data_interim/phase3_partner_type_profile.feather and
data_interim/phase3_full_superclass_profile.feather.

Usage:  python scripts/run_phase3_data.py && python scripts/run_phase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fruresidual.config import load_config
from fruresidual.eval.enrichment import category_enrichment
from fruresidual.features.atlas import load_neurotransmitters
from fruresidual.features.clustering import (
    build_knn_graph,
    build_profile_matrix,
    cluster_enrichment,
    label_permutation_null,
    leiden_cluster,
)
from fruresidual.features.fru_dsx_join import (
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)
from fruresidual.features.motifs import (
    check_degree_confound,
    identify_mixed_strata,
    partner_type_enrichment,
    partner_type_enrichment_degree_adjusted,
)
from fruresidual.viz.phase3 import plot_cluster_scatter, plot_motif_enrichment

MIN_GROUP_SIZE = 20
MIN_MOTIF_NEURONS = 10
# clustering.py's own docstring says not to trust one resolution blindly
# on real data -- swept explicitly rather than picking one arbitrary
# value (an earlier version of this script did exactly that, with
# resolution=0.1 never validated against real data; fixed 2026-09-18
# during a full-project audit).
CLUSTER_RESOLUTION_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.3]
# Permutations for the empirical cluster null. 1000 bounds the smallest
# reportable p-value at ~1/1001, which is ample against alpha=0.05.
N_PERMUTATIONS = 1000

# Annotation dimensions carried over from Phase 1. Phase 1 tested these
# MARGINALLY (whole dimorphic population at once), which predates the
# ol_intrinsic saturation discovery -- so they carried the same regional
# confound the motif/cluster analyses were fixed for. They are re-tested
# here WITHIN each mixed stratum instead.
ANNOTATION_DIMENSIONS = ["itoleeHl", "trumanHl", "somaNeuromere", "consensus_nt", "somaSide"]

# "Not yet assigned a lineage" buckets, not real biological categories.
# Both were the single largest hit in their dimension in Phase 1
# (putative_primary n=74, TBD n=44), and a neuron missing BOTH a fru/dsx
# call and a confident lineage call plausibly just means "under-annotated
# neuron" rather than "biologically distinct lineage" -- exactly the
# base-rate confound docs/phase1_results.md flagged and asked to be
# resolved before these were trusted. Excluded from the shortlist.
PLACEHOLDER_LABELS = frozenset({"putative_primary", "TBD"})


def main() -> None:
    cfg = load_config()
    rule = cfg.fru_dsx_aggregation
    figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures" / "phase3"

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, rule)
    residual_body_ids = set(dimorphic.loc[~dimorphic["has_fru_dsx"], "bodyId"])
    print(f"dimorphic/male-specific: {len(dimorphic)}, residual: {len(residual_body_ids)}")

    # consensus_nt lives in a separate bulk file, same join run_phase1 does
    # -- needed for the stratified annotation-dimension re-test below.
    dimorphic = dimorphic.merge(load_neurotransmitters(cfg), on="bodyId", how="left")

    mixed_strata = identify_mixed_strata(dimorphic, residual_body_ids, min_group_size=MIN_GROUP_SIZE)
    print(f"\nsuperclasses with >= {MIN_GROUP_SIZE} of both residual and explained "
          f"(the only ones where a within-region comparison is non-tautological):")
    print(mixed_strata.to_string(index=False))
    excluded = set(dimorphic["superclass"].unique()) - set(mixed_strata["superclass"])
    print(f"\nexcluded (saturated or too small): {sorted(str(x) for x in excluded)}")

    partner_type_profile = pd.read_feather(cfg.data_interim / "phase3_partner_type_profile.feather")
    superclass_profile = pd.read_feather(cfg.data_interim / "phase3_full_superclass_profile.feather")

    all_motif_results, all_cluster_results, shortlist, degree_confound_results = [], [], [], []
    resolution_sweep_results = []
    all_annotation_results = []
    permutation_null_results = []

    for _, stratum_row in mixed_strata.iterrows():
        sc = stratum_row["superclass"]
        stratum_ids = set(dimorphic.loc[dimorphic["superclass"] == sc, "bodyId"])
        stratum_residual = residual_body_ids & stratum_ids
        print(f"\n{'='*60}\nSTRATUM: {sc} (n={len(stratum_ids)}, "
              f"residual={len(stratum_residual)}, explained={len(stratum_ids) - len(stratum_residual)})\n{'='*60}")

        # --- degree confound check (does NOT control the test below --
        # reported so the motif list is read with the right caveat) ---
        degree_check = check_degree_confound(partner_type_profile, stratum_residual, stratum_ids)
        degree_check["superclass"] = sc
        degree_confound_results.append(degree_check)
        flag = "CONFOUNDED" if degree_check["degree_confounded"] else "not detected"
        print(f"  degree confound check: residual mean degree="
              f"{degree_check['residual_mean_degree']:.0f}, explained="
              f"{degree_check['explained_mean_degree']:.0f}, p="
              f"{degree_check['degree_difference_p_value']:.2e} -> {flag}")

        # --- motif enrichment within this stratum ---
        stratum_motif_profile = partner_type_profile[partner_type_profile["bodyId"].isin(stratum_ids)]
        motif_result = partner_type_enrichment(
            stratum_motif_profile, stratum_residual, stratum_ids, min_neurons=MIN_MOTIF_NEURONS
        )
        motif_result["superclass"] = sc
        sig_motifs = motif_result[motif_result["significant"]]

        # --- degree-adjusted motif enrichment (the one that counts) ---
        # The marginal test above cannot distinguish "residual neurons
        # prefer partner type X" from "residual neurons connect to more
        # of everything". This holds degree constant; only motifs that
        # survive it reach the shortlist.
        adjusted = partner_type_enrichment_degree_adjusted(
            stratum_motif_profile, stratum_residual, stratum_ids, min_neurons=MIN_MOTIF_NEURONS
        )
        adjusted["superclass"] = sc
        adjusted = adjusted.rename(columns={
            "odds_ratio": "odds_ratio_degree_adjusted", "p_value": "p_value_degree_adjusted",
            "p_adj": "p_adj_degree_adjusted", "significant": "significant_degree_adjusted",
        })
        motif_result = motif_result.merge(
            adjusted[["category", "odds_ratio_degree_adjusted", "p_value_degree_adjusted",
                      "p_adj_degree_adjusted", "significant_degree_adjusted", "converged"]],
            on="category", how="left",
        )
        all_motif_results.append(motif_result)
        sig_adjusted = motif_result[motif_result["significant_degree_adjusted"].fillna(False)]

        print(f"  motifs: {len(sig_motifs)}/{len(motif_result)} significant (marginal, degree-CONFOUNDED)"
              + (" <- inflated by the degree difference above" if degree_check["degree_confounded"] else ""))
        print(f"  motifs: {len(sig_adjusted)}/{len(motif_result)} significant after holding degree constant"
              f"  ({100*len(sig_adjusted)/max(len(sig_motifs),1):.1f}% of the marginal hits survive)")
        if len(sig_adjusted):
            print(sig_adjusted.head(10)[["category", "n_in_subset", "pct_of_subset",
                                         "odds_ratio_degree_adjusted", "p_adj_degree_adjusted"]]
                  .to_string(index=False))
        for _, row in sig_adjusted.iterrows():
            shortlist.append({
                "signature_type": f"connectivity_motif[{sc}]", "category": row["category"],
                "pct_of_residual": row["pct_of_subset"],
                "odds_ratio": row["odds_ratio_degree_adjusted"],
                "p_adj": row["p_adj_degree_adjusted"],
                "degree_confounded": False,  # survived explicit degree adjustment
            })

        # --- Phase-1 annotation dimensions, re-tested WITHIN this stratum ---
        stratum_rows = dimorphic[dimorphic["bodyId"].isin(stratum_ids)]
        stratum_residual_mask = stratum_rows["bodyId"].isin(stratum_residual)
        for dim in ANNOTATION_DIMENSIONS:
            if dim not in stratum_rows.columns:
                continue
            sub = stratum_rows.loc[stratum_residual_mask, dim].dropna()
            bg = stratum_rows[dim].dropna()
            if sub.empty or bg.empty or bg.nunique() < 2:
                continue
            ann = category_enrichment(sub, bg)
            ann["superclass"] = sc
            ann["dimension"] = dim
            ann["is_placeholder"] = ann["category"].isin(PLACEHOLDER_LABELS)
            all_annotation_results.append(ann)
            for _, row in ann[ann["significant"] & ~ann["is_placeholder"]].iterrows():
                shortlist.append({
                    "signature_type": f"annotation_{dim}[{sc}]", "category": row["category"],
                    "pct_of_residual": row["pct_of_subset"], "odds_ratio": row["odds_ratio"],
                    "p_adj": row["p_adj"],
                    # Region-stratified now (unlike Phase 1's marginal
                    # version), but these are annotation labels rather
                    # than connectivity, so the degree check does not
                    # apply -- left None rather than claimed either way.
                    "degree_confounded": None,
                })

        # --- clustering within this stratum: resolution sensitivity sweep ---
        stratum_superclass_profile = superclass_profile[superclass_profile["bodyId"].isin(stratum_ids)]
        matrix = build_profile_matrix(stratum_superclass_profile, "partner_superclass")
        if len(matrix) < 30:
            print(f"  skipping clustering: stratum too small ({len(matrix)} neurons with profile data)")
            continue
        graph = build_knn_graph(matrix, k=min(15, len(matrix) - 1))

        sweep_hits = 0
        best_result, best_labels, best_res = None, None, None
        for res in CLUSTER_RESOLUTION_SWEEP:
            labels_r = leiden_cluster(graph, resolution=res, random_state=0)
            result_r = cluster_enrichment(labels_r, stratum_residual)
            n_sig_r = int(result_r["significant"].sum())
            resolution_sweep_results.append({
                "superclass": sc, "resolution": res, "n_clusters": int(labels_r.nunique()),
                "n_significant": n_sig_r,
            })
            if n_sig_r > 0:
                sweep_hits += 1
                if best_result is None:  # keep the smallest (least-smoothed) resolution with a hit
                    best_result, best_labels, best_res = result_r, labels_r, res

        robust = sweep_hits >= (len(CLUSTER_RESOLUTION_SWEEP) + 1) // 2  # majority of swept resolutions
        print(f"  clusters: significant residual enrichment found at {sweep_hits}/{len(CLUSTER_RESOLUTION_SWEEP)} "
              f"swept resolutions -> {'ROBUST' if robust else 'not robust, excluded from shortlist'}")

        # Empirical null for any stratum whose cluster result is robust.
        # The BH p-value from cluster_enrichment assumes cluster
        # assignment is independent of the residual labels, which it is
        # not -- clusters come from the same connectivity data and the
        # resolution was chosen by a sweep. Permuting the labels over the
        # fixed graph gives a null that respects all of that.
        if robust and best_res is not None:
            null = label_permutation_null(
                graph, stratum_residual, resolution=best_res,
                n_permutations=N_PERMUTATIONS, random_state=0,
            )
            null["superclass"] = sc
            permutation_null_results.append(null)
            print(f"  permutation null ({null['n_permutations']} label shuffles, "
                  f"graph and clusters held fixed):")
            print(f"    observed max odds ratio = {null['observed_max_odds_ratio']:.2f}, "
                  f"null mean = {null['null_mean']:.2f}, 95th pct = {null['null_p95']:.2f}")
            print(f"    empirical p = {null['empirical_p_value']:.4f} "
                  f"({null['n_null_ge_observed']}/{null['n_permutations']} shuffles reached it)")

        if best_result is not None:
            best_result["superclass"] = sc
            best_result["resolution"] = best_res
            all_cluster_results.append(best_result)
            labels = best_labels  # for the scatter plot below
            if robust:
                for _, row in best_result[best_result["significant"]].iterrows():
                    shortlist.append({
                        "signature_type": f"connectivity_cluster[{sc}, res={best_res}]",
                        "category": f"cluster_{row['category']}",
                        "pct_of_residual": row["pct_of_subset"], "odds_ratio": row["odds_ratio"],
                        "p_adj": row["p_adj"],
                        # clustering uses L2-normalized profile vectors
                        # (mitigates but doesn't rule out a degree effect)
                        # and single-label enrichment (not the same
                        # multi-label bug as motifs) -- not directly
                        # comparable to the motif check, left unflagged
                        # rather than mislabeled either way.
                        "degree_confounded": None,
                    })
        else:
            labels = leiden_cluster(graph, resolution=CLUSTER_RESOLUTION_SWEEP[0], random_state=0)
        plot_cluster_scatter(
            matrix, labels, stratum_residual, figures_dir / f"cluster_scatter_{sc}.png"
        )

    motif_all = pd.concat(all_motif_results, ignore_index=True) if all_motif_results else pd.DataFrame()
    motif_all.to_csv(cfg.data_processed / "phase3_motif_enrichment_stratified.csv", index=False)
    cluster_all = pd.concat(all_cluster_results, ignore_index=True) if all_cluster_results else pd.DataFrame()
    cluster_all.to_csv(cfg.data_processed / "phase3_cluster_enrichment_stratified.csv", index=False)
    pd.DataFrame(resolution_sweep_results).to_csv(
        cfg.data_processed / "phase3_cluster_resolution_sweep.csv", index=False
    )
    with open(cfg.data_processed / "phase3_degree_confound_check.json", "w") as f:
        json.dump(degree_confound_results, f, indent=2)
    confounded_strata = {d["superclass"] for d in degree_confound_results if d["degree_confounded"]}
    if len(motif_all):
        plot_motif_enrichment(
            motif_all, figures_dir / "motif_enrichment_stratified.png",
            degree_confounded_strata=confounded_strata,
        )
        # The combined top-20-by-p-value chart above is dominated by
        # whichever stratum has the most statistical power (cb_intrinsic,
        # n=1551) -- which also happens to be degree-confounded. A clean,
        # non-confounded result in a smaller stratum (ascending_neuron)
        # would be invisible there even though it's the more trustworthy
        # finding, so give unconfounded strata their own spotlight.
        unconfounded = motif_all[~motif_all["superclass"].isin(confounded_strata)]
        if len(unconfounded):
            plot_motif_enrichment(
                unconfounded, figures_dir / "motif_enrichment_unconfounded_only.png",
                degree_confounded_strata=set(),
            )

    annotation_all = pd.concat(all_annotation_results, ignore_index=True) if all_annotation_results else pd.DataFrame()
    if len(annotation_all):
        annotation_all.to_csv(
            cfg.data_processed / "phase3_annotation_enrichment_stratified.csv", index=False
        )
    print(f"\n{'='*60}")
    print("Phase-1 annotation dimensions, re-tested WITHIN strata")
    print(f"{'='*60}")
    if len(annotation_all):
        sig_ann = annotation_all[annotation_all["significant"]]
        print(f"{len(sig_ann)}/{len(annotation_all)} category-in-stratum tests significant "
              f"(Phase 1's marginal versions gave 30 across the whole population)")
        if len(sig_ann):
            print(sig_ann[["superclass", "dimension", "category", "n_in_subset",
                           "pct_of_subset", "odds_ratio", "p_adj"]].to_string(index=False))
        n_placeholder = int(annotation_all["is_placeholder"].sum())
        print(f"\nplaceholder categories excluded from the shortlist: {n_placeholder} "
              f"({sorted(PLACEHOLDER_LABELS)})")
    else:
        print("none testable")

    shortlist_df = pd.DataFrame(shortlist).sort_values("p_adj") if shortlist else pd.DataFrame()
    shortlist_df.to_csv(cfg.data_processed / "phase3_candidate_shortlist.csv", index=False)

    print(f"\n{'='*60}")
    print(f"CONSOLIDATED SHORTLIST (region-stratified motifs/clusters + Phase-1 non-regional "
          f"dimensions): {len(shortlist_df)} candidates")
    print(f"{'='*60}")
    if len(shortlist_df):
        print(shortlist_df.head(30).to_string(index=False))
    else:
        print("No candidates survived stratified testing -- see docs/phase3_results.md for what that means.")

    summary = {
        "n_dimorphic": len(dimorphic),
        "n_residual": len(residual_body_ids),
        "mixed_strata": mixed_strata.to_dict("records"),
        "excluded_strata": sorted(str(x) for x in excluded),
        "n_motif_candidates_significant_stratified": int(motif_all["significant"].sum()) if len(motif_all) else 0,
        "n_cluster_candidates_significant_stratified": int(cluster_all["significant"].sum()) if len(cluster_all) else 0,
        "degree_confound_by_stratum": degree_confound_results,
        "cluster_resolution_sweep": resolution_sweep_results,
        "cluster_permutation_null": permutation_null_results,
        "shortlist_size": len(shortlist_df),
    }
    with open(cfg.data_processed / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved figures to", figures_dir)
    print("Saved shortlist to", cfg.data_processed / "phase3_candidate_shortlist.csv")


if __name__ == "__main__":
    main()
