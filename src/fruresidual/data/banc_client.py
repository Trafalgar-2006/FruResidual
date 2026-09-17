"""BANC access — independent female CNS, used for Phase 4 cross-dataset
validation (see docs/project_plan.md section b.3). No CAVE auth needed
for the cached per-neuron meta table used here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Populated once the exact public URL for banc_888_meta.feather is confirmed
# during Phase 4 setup (see `bancr::banc_meta_create_cache()` for the R-side
# equivalent, which is the documented reference route).
BANC_META_URL = ""


def load_banc_meta(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        if not BANC_META_URL:
            raise RuntimeError(
                "BANC_META_URL not set yet — confirm the public feather URL "
                "(see docs/project_plan.md section b.3) before Phase 4."
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_feather(BANC_META_URL).to_feather(cache_path)
    return pd.read_feather(cache_path)
