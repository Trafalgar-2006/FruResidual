"""Phase 3 step 1: fetch the two connectivity profiles run_phase3.py needs
-- the full dimorphic/male-specific population's partner-TYPE profile
(for motif enrichment) and partner-SUPERCLASS profile (for clustering
embeddings). Both are targeted, server-side-aggregated neuPrint queries
over the ~2,368-neuron dimorphic population, not a full-connectome pull
(~1 minute total on a normal connection).

This script didn't originally exist -- the profiles were fetched via
ad-hoc interactive scripts during investigation, which live outside the
repo. Without this, a fresh clone can't reproduce Phase 3 at all
(run_phase3.py would fail immediately on the missing cache files). Added
2026-09-18 during a full-project audit, mirroring the
run_phase2_data.py / run_phase2.py split.

Usage:  python scripts/run_phase3_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruresidual.config import load_config
from fruresidual.data.neuprint_client import (
    fetch_partner_superclass_profile,
    fetch_partner_type_profile,
)
from fruresidual.features.fru_dsx_join import DIMORPHIC_CATEGORIES, load_male_cns_annotations


def main() -> None:
    cfg = load_config()
    cfg.data_interim.mkdir(parents=True, exist_ok=True)

    df = load_male_cns_annotations(cfg, traced_only=True)
    dimorphic = df[df["dimorphism"].isin(DIMORPHIC_CATEGORIES)]
    body_ids = dimorphic["bodyId"].tolist()
    print(f"dimorphic/male-specific population: {len(body_ids)} neurons")

    type_path = cfg.data_interim / "phase3_partner_type_profile.feather"
    if type_path.exists():
        print(f"skip (already have): {type_path.name}")
    else:
        print("fetching partner-type profile (fine-grained, for motif enrichment)...")
        profile = fetch_partner_type_profile(cfg, body_ids)
        profile.to_feather(type_path)
        print(f"  -> {len(profile)} rows, {profile['partner_type'].nunique()} distinct types, saved to {type_path}")

    superclass_path = cfg.data_interim / "phase3_full_superclass_profile.feather"
    if superclass_path.exists():
        print(f"skip (already have): {superclass_path.name}")
    else:
        print("fetching partner-superclass profile (coarse, for clustering embeddings)...")
        profile = fetch_partner_superclass_profile(cfg, body_ids)
        profile.to_feather(superclass_path)
        print(f"  -> {len(profile)} rows, saved to {superclass_path}")


if __name__ == "__main__":
    main()
