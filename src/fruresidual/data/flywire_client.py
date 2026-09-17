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
