"""Bulk-downloads the MaleCNS v1.0 feather files (no auth needed — public
GCS bucket, see docs/project_plan.md section b.1 route B).

Downloads to a `.part` file and renames on success, so a partial file from
an interrupted run is never mistaken for a complete one. Resumes partial
downloads via HTTP Range instead of restarting from zero — this
connection has been slow and flaky enough (read-timeouts on the ~1GB
connectome-weights file) that resume support isn't optional.

Usage:  python scripts/download_male_cns.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruresidual.config import load_config

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
# Small-to-large order on purpose: the two small files are all Phase 0
# needs, so they should land first. connectome-weights (~1GB) is only
# needed for Phase 2's GNN and can crawl along after on a slow connection.
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]
MAX_RETRIES = 20


def download_with_resume(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            resp = requests.get(url, stream=True, timeout=60, headers=headers)
            if resp.status_code == 416:  # range beyond end -> already complete
                break
            resp.raise_for_status()
            mode = "ab" if resume_from else "wb"
            with open(part, mode) as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            break
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed ({e}); "
                  f"resuming from {part.stat().st_size if part.exists() else 0} bytes")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 30))
    part.rename(dest)


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
        download_with_resume(url, dest)
        print(f"  -> saved to {dest}")


if __name__ == "__main__":
    main()
