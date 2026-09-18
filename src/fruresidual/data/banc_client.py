"""BANC access — independent female CNS, used for Phase 4 cross-dataset
validation (see docs/project_plan.md section b.3).

Two access routes exist, and this project only uses one of them:
- Live connectivity via BANC's CAVE API (`banc.get_caveclient()`) --
  requires per-dataset "view" permission on top of a valid CAVE token;
  confirmed 2026-09-19 that our account has a valid token (it already
  works for FlyWire) but NOT view permission for BANC specifically
  (403 "Missing permission: view for dataset BANC"). This is an
  external, account-level grant this project cannot self-serve, so live
  BANC connectivity queries (the Phase-2-style wiring-difference
  approach) are not available here.
- The public, no-auth bulk metadata table used below -- confirmed via
  direct HEAD request (2026-09-19, 200 OK, no auth) at the GCS path
  bancr's `banc_meta_create_cache()` uses on the R side. This is a
  per-neuron annotation table (188,508 rows x 81 columns), not a
  connectivity/edge table -- it includes BANC's own independent
  `sexually_dimorphic` call and direct neuron-level `malecns_match`
  (MaleCNS bodyId) cross-references, which is what Phase 4 is actually
  built on instead of the originally-planned connectivity comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BANC_META_URL = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
    "compiled_data/banc_888/banc_888_meta.feather"
)


def load_banc_meta(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(BANC_META_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return pd.read_feather(cache_path)
