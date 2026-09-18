"""Phase 2 step 1: build the case/control population and the per-neuron
wiring-difference target. See docs/project_plan.md section (d), PHASE 2,
and src/fruresidual/features/rewiring.py for the scoping rationale.

Usage:  python scripts/run_phase2_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruresidual.config import load_config
from fruresidual.data.flywire_client import (
    fetch_flywire_type_profiles,
    load_flywire_annotations,
)
from fruresidual.data.neuprint_client import fetch_partner_superclass_profile
from fruresidual.features.fru_dsx_join import load_male_cns_annotations
from fruresidual.features.rewiring import (
    cosine_distance,
    malecns_superclass_to_shared,
    profile_vector,
    select_case_control_population,
)

CONTROL_RATIO = 1.0
SAMPLES_PER_TYPE = 15
RANDOM_STATE = 0


def main() -> None:
    cfg = load_config()
    cfg.data_interim.mkdir(parents=True, exist_ok=True)
    cfg.data_processed.mkdir(parents=True, exist_ok=True)

    mcns = load_male_cns_annotations(cfg, traced_only=True)
    fw = load_flywire_annotations(cfg.data_raw / "flywire_annotations.feather")
    fw_types = set(fw["cell_type"].dropna().unique())

    cases, controls = select_case_control_population(
        mcns, flywire_cell_types=fw_types, control_ratio=CONTROL_RATIO, random_state=RANDOM_STATE
    )
    population = pd.concat([cases, controls], ignore_index=True)
    print(f"cases: {len(cases)}, controls: {len(controls)}, total: {len(population)}")

    # --- MaleCNS-side partner-superclass profiles (targeted, fast) ---
    male_profile_cache = cfg.data_interim / "phase2_male_partner_profiles.feather"
    if male_profile_cache.exists():
        male_profiles_long = pd.read_feather(male_profile_cache)
    else:
        male_profiles_long = fetch_partner_superclass_profile(cfg, population["bodyId"].tolist())
        male_profiles_long.to_feather(male_profile_cache)
    print(f"male-side partner profiles: {len(male_profiles_long)} rows for "
          f"{male_profiles_long['bodyId'].nunique()} neurons")

    # --- FlyWire-side type-level profiles (targeted, slower) ---
    # Checkpointed every CHECKPOINT_EVERY types: this connection has
    # already hit a mid-download read-timeout once (connectome-weights),
    # so a 100+-type, tens-of-minutes fetch shouldn't risk losing all
    # progress to one dropped connection. Rerunning this script skips
    # whatever's already cached.
    CHECKPOINT_EVERY = 5
    fw_profile_cache = cfg.data_interim / "phase2_flywire_type_profiles.feather"
    distinct_types = sorted(population["flywireType"].unique())
    if fw_profile_cache.exists():
        fw_profiles_long = pd.read_feather(fw_profile_cache)
        cached_types = set(fw_profiles_long["flywireType"].unique())
        missing = [t for t in distinct_types if t not in cached_types]
    else:
        fw_profiles_long = pd.DataFrame(columns=["flywireType", "partner_super_class", "weight"])
        missing = distinct_types
    if missing:
        print(f"fetching FlyWire connectivity for {len(missing)} types not yet cached...")
        for i in range(0, len(missing), CHECKPOINT_EVERY):
            chunk = missing[i : i + CHECKPOINT_EVERY]
            new_profiles = fetch_flywire_type_profiles(fw, chunk, samples_per_type=SAMPLES_PER_TYPE)
            fw_profiles_long = pd.concat([fw_profiles_long, new_profiles], ignore_index=True)
            fw_profiles_long.to_feather(fw_profile_cache)
            print(f"  checkpointed {min(i + CHECKPOINT_EVERY, len(missing))}/{len(missing)} types")
    print(f"FlyWire-side type profiles: {len(fw_profiles_long)} rows for "
          f"{fw_profiles_long['flywireType'].nunique()} types")

    # --- Build vectors and compute wiring-difference target ---
    # Map MaleCNS partner superclasses to the shared vocabulary once,
    # vectorized, before pivoting -- pivot_table's sum aggfunc then
    # naturally combines e.g. ol_intrinsic + ol_sensory into one "optic"
    # column, instead of remapping per-row in a python loop.
    male_shared_long = male_profiles_long.copy()
    male_shared_long["shared_category"] = malecns_superclass_to_shared(
        male_shared_long["partner_superclass"]
    )
    male_shared_long = male_shared_long.dropna(subset=["shared_category"])
    male_pivot = male_shared_long.pivot_table(
        index="bodyId", columns="shared_category", values="weight", aggfunc="sum", fill_value=0
    )
    fw_pivot = fw_profiles_long.pivot_table(
        index="flywireType", columns="partner_super_class", values="weight", fill_value=0
    )

    rows = []
    for _, neuron in population.iterrows():
        body_id, ftype = neuron["bodyId"], neuron["flywireType"]
        if body_id not in male_pivot.index or ftype not in fw_pivot.index:
            continue
        male_vec = profile_vector(male_pivot.loc[body_id].to_dict())
        fw_vec = profile_vector(fw_pivot.loc[ftype].to_dict())
        wiring_diff = cosine_distance(male_vec, fw_vec)
        rows.append({"bodyId": body_id, "flywireType": ftype, "wiring_difference": wiring_diff,
                      "is_case": neuron["is_case"]})

    targets = pd.DataFrame(rows)
    out_path = cfg.data_processed / "phase2_wiring_targets.feather"
    targets.to_feather(out_path)
    print(f"\nsaved {len(targets)} wiring-difference targets to {out_path}")
    print(targets.groupby("is_case")["wiring_difference"].describe())


if __name__ == "__main__":
    main()
