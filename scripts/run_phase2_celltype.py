"""Phase 2 at CELL-TYPE resolution: does the wiring-difference result
survive a finer partner vocabulary?

The Phase 2 target compares partner profiles over 10 superclass buckets.
That is coarse enough to be blind to the most obvious form of rewiring --
"same broad partner category, entirely different specific partners" --
and it is the largest acknowledged methodological weakness left in the
project (docs/paper.md section 5). This rebuilds the same target over
partner CELL TYPES instead, and reruns the ablation on it.

Run as a parallel analysis rather than a replacement, so the two
resolutions can be compared directly: a conclusion that holds at both is
worth more than one that only holds at the resolution it was designed at.

The shared-vocabulary problem is the whole difficulty. MaleCNS partner
types ("CB1078", "DNp48") and FlyWire cell types are different naming
systems, so they are mapped into one space via MaleCNS's own
`flywireType` field, and then BOTH sides are restricted to the
intersection. That symmetry is not optional: leaving categories in the
vector that only one dataset can express is exactly the bug that made
the original target ~76% artifact (docs/phase2_results.md).

Usage:  python scripts/run_phase2_celltype.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from fruresidual.config import load_config
from fruresidual.data.flywire_client import fetch_flywire_type_profiles, load_flywire_annotations
from fruresidual.data.neuprint_client import fetch_partner_type_profile
from fruresidual.features.atlas import load_neurotransmitters
from fruresidual.features.fru_dsx_join import load_male_cns_annotations
from fruresidual.features.ml_features import build_features
from fruresidual.features.rewiring import select_case_control_population
from fruresidual.models.baselines import evaluate_baselines

RANDOM_STATE = 0
# Fewer samples per type than Phase 2's 15. At cell-type resolution each
# query returns far more partner rows, and the live FlyWire connection
# has hung mid-fetch twice on this project; a smaller per-type budget
# both speeds the pass and shortens the window for a hang.
SAMPLES_PER_TYPE = 8
# One type per checkpoint, not five. A hang loses at most that one type's
# work, and the printed type name says exactly where it stopped -- an
# earlier 5-type chunking gave no visibility and the fetch stalled
# silently on an unknown type for ten minutes.
CHECKPOINT_EVERY = 1
# A partner type has to be seen on enough neurons to carry information;
# below this the profile vector is mostly noise from single connections.
MIN_PARTNER_NEURONS = 5


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1 - np.dot(a, b) / (na * nb))


def main() -> None:
    cfg = load_config()
    cfg.data_interim.mkdir(parents=True, exist_ok=True)
    cfg.data_processed.mkdir(parents=True, exist_ok=True)

    mcns = load_male_cns_annotations(cfg, traced_only=True)
    fw = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
    fw_types = set(fw["cell_type"].dropna().unique())
    cases, controls = select_case_control_population(
        mcns, flywire_cell_types=fw_types, random_state=RANDOM_STATE
    )
    population = pd.concat([cases, controls], ignore_index=True)
    print(f"population: {len(population)} ({int(population['is_case'].sum())} cases)")

    # --- MaleCNS side, at partner-type resolution ---
    male_path = cfg.data_interim / "phase2ct_male_partner_type_profiles.feather"
    if male_path.exists():
        male_long = pd.read_feather(male_path)
        print(f"male-side partner-type profiles (cached): {len(male_long)} rows")
    else:
        print("fetching MaleCNS partner-TYPE profiles...")
        male_long = fetch_partner_type_profile(cfg, population["bodyId"].tolist())
        male_long.to_feather(male_path)
        print(f"  -> {len(male_long)} rows, {male_long['partner_type'].nunique()} types")

    # Map MaleCNS partner types into FlyWire's vocabulary. A partner's
    # flywireType is the only bridge between the two naming systems.
    # Comma-containing values are ambiguous multi-type matches and are
    # dropped, the same rule the rest of the project uses.
    type_to_fw = (
        mcns.loc[mcns["flywireType"].notna() & ~mcns["flywireType"].str.contains(",", na=False),
                 ["type", "flywireType"]]
        .dropna().drop_duplicates("type").set_index("type")["flywireType"]
    )
    male_long["shared_type"] = male_long["partner_type"].map(type_to_fw)
    mapped = male_long["shared_type"].notna()
    print(f"MaleCNS partner rows mappable to a FlyWire type: "
          f"{int(mapped.sum())}/{len(male_long)} ({100*mapped.mean():.1f}%)")
    male_long = male_long[mapped]

    # --- FlyWire side, at partner-cell_type resolution ---
    fw_path = cfg.data_interim / "phase2ct_flywire_celltype_profiles.feather"
    distinct_types = sorted(population["flywireType"].unique())
    if fw_path.exists():
        fw_long = pd.read_feather(fw_path)
        cached = set(fw_long["flywireType"].unique())
        missing = [t for t in distinct_types if t not in cached]
    else:
        fw_long = pd.DataFrame(columns=["flywireType", "partner_cell_type", "weight"])
        missing = distinct_types
    failed: list[str] = []
    if missing:
        print(f"fetching FlyWire partner-CELL-TYPE profiles for {len(missing)} types "
              f"(live queries, one checkpoint per type)...", flush=True)
        for i in range(0, len(missing), CHECKPOINT_EVERY):
            chunk = missing[i : i + CHECKPOINT_EVERY]
            print(f"  [{i+1}/{len(missing)}] {chunk}", flush=True)
            try:
                new = fetch_flywire_type_profiles(
                    fw, chunk, samples_per_type=SAMPLES_PER_TYPE,
                    partner_group_col="cell_type",
                )
            except Exception as e:
                # One flaky type must not abort a fetch this long. Record
                # it and move on; a rerun retries only what is missing.
                print(f"      FAILED ({type(e).__name__}: {e}) -- skipping", flush=True)
                failed.extend(chunk)
                continue
            fw_long = pd.concat([fw_long, new], ignore_index=True)
            fw_long.to_feather(fw_path)
        if failed:
            print(f"  {len(failed)} type(s) failed and were skipped: {failed[:10]}")
    print(f"FlyWire-side profiles: {len(fw_long)} rows, "
          f"{fw_long['partner_cell_type'].nunique()} distinct partner types")

    # --- Shared vocabulary, enforced symmetrically ---
    male_counts = male_long.groupby("shared_type")["bodyId"].nunique()
    male_vocab = set(male_counts[male_counts >= MIN_PARTNER_NEURONS].index)
    fw_counts = fw_long.groupby("partner_cell_type")["flywireType"].nunique()
    fw_vocab = set(fw_counts[fw_counts >= MIN_PARTNER_NEURONS].index)
    shared = sorted(male_vocab & fw_vocab)
    print(f"\nshared partner vocabulary: {len(shared)} cell types "
          f"(MaleCNS-side {len(male_vocab)}, FlyWire-side {len(fw_vocab)})")
    if len(shared) < 20:
        raise SystemExit(
            f"Only {len(shared)} shared partner types -- too few for a meaningful "
            "profile comparison. The flywireType bridge is probably not resolving; "
            "check the mapping rate printed above."
        )

    male_pivot = (male_long[male_long["shared_type"].isin(shared)]
                  .pivot_table(index="bodyId", columns="shared_type", values="weight",
                               aggfunc="sum", fill_value=0))
    fw_pivot = (fw_long[fw_long["partner_cell_type"].isin(shared)]
                .pivot_table(index="flywireType", columns="partner_cell_type", values="weight",
                             aggfunc="sum", fill_value=0))
    male_pivot = male_pivot.reindex(columns=shared, fill_value=0)
    fw_pivot = fw_pivot.reindex(columns=shared, fill_value=0)

    rows = []
    for _, n in population.iterrows():
        b, ftype = n["bodyId"], n["flywireType"]
        if b not in male_pivot.index or ftype not in fw_pivot.index:
            continue
        d = _cosine_distance(male_pivot.loc[b].to_numpy(float), fw_pivot.loc[ftype].to_numpy(float))
        rows.append({"bodyId": b, "flywireType": ftype, "wiring_difference": d,
                     "is_case": bool(n["is_case"])})
    targets = pd.DataFrame(rows)
    n_unscorable = int(targets["wiring_difference"].isna().sum())
    if n_unscorable:
        print(f"dropping {n_unscorable} neurons with no comparable shared-type connectivity")
        targets = targets.dropna(subset=["wiring_difference"]).reset_index(drop=True)
    targets.to_feather(cfg.data_processed / "phase2ct_wiring_targets.feather")
    print(f"scored {len(targets)} neurons at cell-type resolution")

    case = targets.loc[targets["is_case"], "wiring_difference"]
    control = targets.loc[~targets["is_case"], "wiring_difference"]
    u, p_sep = mannwhitneyu(case, control, alternative="greater")
    rank_biserial = 2 * u / (len(case) * len(control)) - 1
    print(f"\ncases mean={case.mean():.5f} median={case.median():.5f}")
    print(f"controls mean={control.mean():.5f} median={control.median():.5f}")
    print(f"Mann-Whitney (cases > controls): p={p_sep:.3e}, effect size={rank_biserial:+.3f}")

    # --- Same ablation, on the finer target ---
    nt = load_neurotransmitters(cfg)
    pop = population.merge(nt, on="bodyId", how="left")
    male_super = pd.read_feather(cfg.data_interim / "phase2_male_partner_profiles.feather")
    X_without = build_features(pop, male_super, include_fru_dsx=False)
    X_with = build_features(pop, male_super, include_fru_dsx=True)
    body_ids = X_without["bodyId"]
    y = targets.set_index("bodyId")["wiring_difference"].reindex(body_ids)
    keep = y.notna().to_numpy()
    y = y[keep].reset_index(drop=True)

    res_without = evaluate_baselines(X_without.drop(columns=["bodyId"])[keep], y, random_state=RANDOM_STATE)
    res_without["ablation"] = "without_fru_dsx"
    res_with = evaluate_baselines(X_with.drop(columns=["bodyId"])[keep], y, random_state=RANDOM_STATE)
    res_with["ablation"] = "with_fru_dsx"
    ablation = pd.concat([res_without, res_with], ignore_index=True)
    ablation.to_csv(cfg.data_processed / "phase2ct_ablation_table.csv", index=False)
    print("\n=== ablation at cell-type resolution ===")
    print(ablation.to_string(index=False))

    summary = {
        "resolution": "partner_cell_type",
        "n_shared_partner_types": len(shared),
        "n_scored": len(targets),
        "n_cases": int(targets["is_case"].sum()),
        "n_controls": int((~targets["is_case"]).sum()),
        "case_mean": float(case.mean()), "control_mean": float(control.mean()),
        "case_median": float(case.median()), "control_median": float(control.median()),
        "separation_p_value": float(p_sep),
        "separation_rank_biserial": float(rank_biserial),
        "ablation_table": ablation.to_dict("records"),
    }
    out = cfg.data_processed / "phase2ct_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
