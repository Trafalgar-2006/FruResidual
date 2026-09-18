"""FlyWire/FAFB access: the curated flywire_annotations table is the source
of truth for dimorphism/fru_dsx labels — prefer it over Codex exports
(see docs/project_plan.md section b.2, "Codex vs curated annotations").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ANNOTATIONS_RAW_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)


def load_flywire_annotations(cache_path: Path) -> pd.DataFrame:
    """Loads the curated flywire_annotations TSV, caching it to cache_path.

    Columns of interest for this project: `dimorphism`, `fru_dsx`,
    `matching_notes`, `supertype`.
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(ANNOTATIONS_RAW_URL, sep="\t", low_memory=False)
        df.to_feather(cache_path)
    return pd.read_feather(cache_path)


def fetch_flywire_type_profiles(
    flywire_annotations: pd.DataFrame,
    flywire_types: list[str],
    samples_per_type: int = 15,
    random_state: int = 0,
) -> pd.DataFrame:
    """For each cell type in `flywire_types`, samples up to
    `samples_per_type` FlyWire neurons of that type, fetches their
    connectivity (fafbseg.flywire.get_connectivity), and aggregates
    partner connection weight by partner super_class -- the FlyWire-side
    input to the Phase 2 wiring-difference target
    (fruresidual.features.rewiring). Type-level, not neuron-level,
    because individual male<->female neuron correspondence isn't defined
    across two different animals; sampling per type (rather than pulling
    every member) keeps the query budget bounded regardless of type size
    (some types have 100+ members -- see docs/phase2_results.md).

    Returns long format: (flywireType, partner_super_class, weight).
    """
    import fafbseg

    root_col, type_col, sc_col = "root_id", "cell_type", "super_class"
    sc_lookup = flywire_annotations.set_index(root_col)[sc_col]

    rows = []
    for ftype in flywire_types:
        members = flywire_annotations.loc[flywire_annotations[type_col] == ftype, root_col]
        if members.empty:
            continue
        sample = members.sample(n=min(len(members), samples_per_type), random_state=random_state).tolist()
        conn = fafbseg.flywire.get_connectivity(sample, proofread_only=False, progress=False)
        if conn.empty:
            continue
        sample_set = set(sample)
        out_partners = conn.loc[conn["pre"].isin(sample_set), ["post", "weight"]].rename(columns={"post": "partner"})
        in_partners = conn.loc[conn["post"].isin(sample_set), ["pre", "weight"]].rename(columns={"pre": "partner"})
        partners = pd.concat([out_partners, in_partners], ignore_index=True)
        partners["partner_super_class"] = partners["partner"].map(sc_lookup)
        agg = partners.groupby("partner_super_class")["weight"].sum()
        for sc, weight in agg.items():
            if pd.notna(sc):
                rows.append({"flywireType": ftype, "partner_super_class": sc, "weight": float(weight)})

    return pd.DataFrame(rows)
