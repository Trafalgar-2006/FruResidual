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


CODEX_FRU_DSX_GENES = frozenset({"fruitless", "doublesex", "fru", "dsx"})

# Codex's export headers are not pinned by any public schema doc and have
# changed between releases, so each field is matched against candidates
# (case-insensitive) rather than assumed.
_CODEX_COLUMN_CANDIDATES = {
    "cell_type": ["cell_type", "type", "cell type", "primary_type"],
    "super_class": ["super_class", "superclass", "super class", "class"],
    "gene": ["gene", "genes", "gene_expression", "marker", "markers"],
}


def load_codex_annotations(path: Path) -> pd.DataFrame:
    """Loads a Codex 'Download Data' export and normalizes it into the
    shape fruresidual.features.annotation_crosscheck expects:
    (cell_type, super_class, fru_dsx).

    Why this exists: the `fru_dsx` column in the curated
    flywire_annotations table was derived with reference to MaleCNS --
    verified empirically, it never marks a neuron positive where MaleCNS
    marked it negative, so it cannot adjudicate whether the residual is
    biology or MaleCNS annotation coverage (see
    docs/annotation_crosscheck_results.md). Codex's `gene` field comes
    from a different annotation lineage; its README explicitly warns its
    labels "likely diverge" from the curated table, which for this one
    purpose is exactly what is wanted.

    Accepts .csv/.csv.gz/.tsv/.feather. Column names are auto-detected
    because Codex's export headers are not stable across releases; if a
    required field cannot be found the error lists the columns that ARE
    present rather than failing on a KeyError deep in a groupby.
    """
    if path.suffix == ".feather":
        raw = pd.read_feather(path)
    else:
        sep = "\t" if path.suffix in {".tsv", ".tab"} else ","
        raw = pd.read_csv(path, sep=sep, low_memory=False)

    lowered = {str(c).strip().lower(): c for c in raw.columns}
    resolved: dict[str, str] = {}
    for field, candidates in _CODEX_COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in lowered:
                resolved[field] = lowered[cand]
                break
        if field not in resolved:
            raise ValueError(
                f"Could not find a '{field}' column in {path.name}. Tried {candidates}. "
                f"Columns present: {sorted(raw.columns)}. "
                "If Codex renamed it, add the new name to _CODEX_COLUMN_CANDIDATES."
            )

    gene = raw[resolved["gene"]].astype("string").str.strip().str.lower()
    # A neuron's gene field can list several markers; positive if ANY of
    # them is fru/dsx. Same "any" logic the type-level rollup uses, and
    # for the same reason: it biases toward finding fru/dsx on the
    # residual, i.e. against this analysis's own hypothesis.
    is_fru_dsx = gene.fillna("").apply(
        lambda g: any(tok.strip() in CODEX_FRU_DSX_GENES for tok in g.replace(";", ",").split(","))
    )

    out = pd.DataFrame({
        "cell_type": raw[resolved["cell_type"]],
        "super_class": raw[resolved["super_class"]],
        # crosscheck treats notna() as positive, so encode negatives as NA
        "fru_dsx": gene.where(is_fru_dsx, other=pd.NA),
    })
    return out[out["cell_type"].notna()].reset_index(drop=True)


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
