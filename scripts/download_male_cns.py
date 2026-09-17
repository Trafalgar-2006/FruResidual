"""Bulk-downloads the MaleCNS v1.0 feather files (no auth needed — public
GCS bucket, see docs/project_plan.md section b.1 route B).

Usage:  python scripts/download_male_cns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruresidual.config import load_config  # noqa: E402

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
# Small-to-large order on purpose: the two small files are all Phase 0
# needs, so they should land first. connectome-weights (~1GB) is only
# needed for Phase 2's GNN and can crawl along after on a slow connection.
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]


def main() -> None:
    cfg = load_config()
    cfg.data_raw.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = cfg.data_raw / filename
        if dest.exists():
            print(f"skip (already have): {filename}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"downloading {filename} ...")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  -> saved to {dest}")


if __name__ == "__main__":
    main()
