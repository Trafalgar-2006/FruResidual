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

Requires the cached profiles fetched during interactive investigation:
data_interim/phase3_partner_type_profile.feather and
data_interim/phase3_full_superclass_profile.feather.

Usage:  python scripts/run_phase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fruresidual.config import load_config  # noqa: E402
from fruresidual.features.clustering import (  # noqa: E402
    build_knn_graph,
    build_profile_matrix,
    cluster_enrichment,
    leiden_cluster,
)
from fruresidual.features.fru_dsx_join import (  # noqa: E402
    DIMORPHIC_CATEGORIES,
    has_fru_dsx,
    load_male_cns_annotations,
)
from fruresidual.features.motifs import (  # noqa: E402
    check_degree_confound,
    identify_mixed_strata,
    partner_type_enrichment,
)
from fruresidual.viz.phase3 import plot_cluster_scatter, plot_motif_enrichment  # noqa: E402

RULE = "any_positive"
MIN_GROUP_SIZE = 20
MIN_MOTIF_NEURONS = 10
CLUSTER_RESOLUTION = 0.1  # see clustering.py docstring for why not 1.0


def main() -> None:
    cfg = load_config()
    figures_dir = Path(__file__).resolve().parents[1] / "reports" / "figures" / "phase3"

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)].copy()
    dimorphic["has_fru_dsx"] = has_fru_dsx(dimorphic, RULE)
    residual_body_ids = set(dimorphic.loc[~dimorphic["has_fru_dsx"], "bodyId"])
    print(f"dimorphic/male-specific: {len(dimorphic)}, residual: {len(residual_body_ids)}")

    mixed_strata = identify_mixed_strata(dimorphic, residual_body_ids, min_group_size=MIN_GROUP_SIZE)
    print(f"\nsuperclasses with >= {MIN_GROUP_SIZE} of both residual and explained "
          f"(the only ones where a within-region comparison is non-tautological):")
    print(mixed_strata.to_string(index=False))
    excluded = set(dimorphic["superclass"].unique()) - set(mixed_strata["superclass"])
    print(f"\nexcluded (saturated or too small): {sorted(str(x) for x in excluded)}")

    partner_type_profile = pd.read_feather(cfg.data_interim / "phase3_partner_type_profile.feather")
    superclass_profile = pd.read_feather(cfg.data_interim / "phase3_full_superclass_profile.feather")

    all_motif_results, all_cluster_results, shortlist, degree_confound_results = [], [], [], []

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
        all_motif_results.append(motif_result)
        sig_motifs = motif_result[motif_result["significant"]]
        print(f"  motifs: {len(sig_motifs)}/{len(motif_result)} significant "
              f"(of {stratum_motif_profile['partner_type'].nunique()} raw distinct types)"
              + (" -- READ WITH DEGREE-CONFOUND CAVEAT ABOVE" if degree_check["degree_confounded"] else ""))
        if len(sig_motifs):
            print(sig_motifs.head(10)[["category", "n_in_subset", "pct_of_subset", "odds_ratio", "p_adj"]]
                  .to_string(index=False))
        for _, row in sig_motifs.iterrows():
            shortlist.append({
                "signature_type": f"connectivity_motif[{sc}]", "category": row["category"],
                "pct_of_residual": row["pct_of_subset"], "odds_ratio": row["odds_ratio"], "p_adj": row["p_adj"],
                "degree_confounded": degree_check["degree_confounded"],
            })

        # --- clustering within this stratum ---
        stratum_superclass_profile = superclass_profile[superclass_profile["bodyId"].isin(stratum_ids)]
        matrix = build_profile_matrix(stratum_superclass_profile, "partner_superclass")
        if len(matrix) < 30:
            print(f"  skipping clustering: stratum too small ({len(matrix)} neurons with profile data)")
            continue
        graph = build_knn_graph(matrix, k=min(15, len(matrix) - 1))
        labels = leiden_cluster(graph, resolution=CLUSTER_RESOLUTION, random_state=0)
        cluster_result = cluster_enrichment(labels, stratum_residual)
        cluster_result["superclass"] = sc
        all_cluster_results.append(cluster_result)
        sig_clusters = cluster_result[cluster_result["significant"]]
        print(f"  clusters: {labels.nunique()} found, {len(sig_clusters)} significantly enriched for residual")
        for _, row in sig_clusters.iterrows():
            shortlist.append({
                "signature_type": f"connectivity_cluster[{sc}]", "category": f"cluster_{row['category']}",
                "pct_of_residual": row["pct_of_subset"], "odds_ratio": row["odds_ratio"], "p_adj": row["p_adj"],
                # clustering uses L2-normalized profile vectors (mitigates
                # but doesn't rule out a degree effect) and single-label
                # enrichment (not the same multi-label bug as motifs) --
                # not directly comparable to the motif check above, so
                # left unflagged rather than mislabeled either way.
                "degree_confounded": None,
            })
        plot_cluster_scatter(
            matrix, labels, stratum_residual, figures_dir / f"cluster_scatter_{sc}.png"
        )

    motif_all = pd.concat(all_motif_results, ignore_index=True) if all_motif_results else pd.DataFrame()
    motif_all.to_csv(cfg.data_processed / "phase3_motif_enrichment_stratified.csv", index=False)
    cluster_all = pd.concat(all_cluster_results, ignore_index=True) if all_cluster_results else pd.DataFrame()
    cluster_all.to_csv(cfg.data_processed / "phase3_cluster_enrichment_stratified.csv", index=False)
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

    # fold in Phase 1's dimensions -- those were already computed marginally
    # (not per-superclass), but hemilineage/neurotransmitter/neuromere are
    # not the SAME confound axis as superclass, so including them as
    # additional (separately caveated) candidate lines is reasonable --
    # flagged explicitly in the writeup, not silently merged as equivalent
    # evidence to the stratified motif/cluster results above.
    for dim in ("itoleeHl", "trumanHl", "somaNeuromere", "consensus_nt", "somaSide"):
        path = cfg.data_processed / f"phase1_enrichment_{dim}.csv"
        if not path.exists():
            continue
        dim_result = pd.read_csv(path)
        for _, row in dim_result[dim_result["significant"]].iterrows():
            shortlist.append({
                "signature_type": f"phase1_marginal_{dim}", "category": row["category"],
                "pct_of_residual": row["pct_of_subset"], "odds_ratio": row["odds_ratio"], "p_adj": row["p_adj"],
                # "marginal" in the signature_type name is the caveat that
                # matters here: these were NOT region-stratified (Phase 1
                # predates discovering the ol_intrinsic saturation issue),
                # so some may carry the same regional-confound risk that
                # motifs/clusters above were corrected for. Not degree-
                # checked either -- neither caveat resolved, both flagged
                # via the signature_type label rather than guessed at.
                "degree_confounded": None,
            })

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
        "shortlist_size": len(shortlist_df),
    }
    with open(cfg.data_processed / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved figures to", figures_dir)
    print("Saved shortlist to", cfg.data_processed / "phase3_candidate_shortlist.csv")


if __name__ == "__main__":
    main()
