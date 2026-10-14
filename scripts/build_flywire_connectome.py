"""Build the signed, weighted FlyWire (female) connectome as a sparse matrix.

Mirrors build_connectome.py so the two sexes are processed identically --
same restriction to proofread cells, same presynaptic-transmitter sign
convention, same output format. Anything that differs between the two
matrices should be biology or dataset scope, not pipeline.

FlyWire is BRAIN-ONLY (FAFB). It has no ventral nerve cord, so TN1, dPR1
and the wing motor neurons simply do not exist here, and neither does
pIP10 (male-specific). Any male/female comparison therefore has to read
out something both datasets contain -- see run_sex_comparison.py.

Source: Zenodo record 10676866, proofread_connections_783.feather.

Usage:  python scripts/build_flywire_connectome.py [--min-weight N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

RAW = Path("C:/CourtshipData/raw")
OUT_DIR = Path("C:/CourtshipData/processed")
ANNOTATIONS = Path("C:/FruResidualData/raw/flywire_annotations.feather")
CONNECTIONS = RAW / "proofread_connections_783.feather"

EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate", "histamine"}
UNKNOWN_SIGN = 0.0

# The published column names have varied between releases, so resolve them
# rather than assume. Guessing wrong here would silently build a transposed
# or mis-weighted matrix.
PRE_CANDIDATES = ["pre_root_id", "pre_pt_root_id", "pre", "source"]
POST_CANDIDATES = ["post_root_id", "post_pt_root_id", "post", "target"]
WEIGHT_CANDIDATES = ["syn_count", "weight", "n_syn", "count"]


def _pick(cols: list[str], candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"could not find the {what} column among {cols}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weight", type=int, default=1)
    args = ap.parse_args()

    if not CONNECTIONS.exists():
        raise SystemExit(f"missing {CONNECTIONS} -- download it from Zenodo 10676866")

    ann = pd.read_feather(ANNOTATIONS)
    # Nodes are the annotated, proofread neurons -- the direct analogue of
    # MaleCNS's traced set.
    body_ids = np.sort(ann["root_id"].to_numpy().astype(np.int64))
    n = len(body_ids)
    print(f"FlyWire annotated neurons (graph nodes): {n:,}")

    nt_map = dict(zip(ann["root_id"].to_numpy(), ann["top_nt"].to_numpy(), strict=False))
    sign = np.full(n, UNKNOWN_SIGN, dtype=np.float32)
    counts = {"excitatory": 0, "inhibitory": 0, "unsigned": 0}
    for i, b in enumerate(body_ids):
        v = nt_map.get(b)
        if v in EXCITATORY:
            sign[i] = 1.0
            counts["excitatory"] += 1
        elif v in INHIBITORY:
            sign[i] = -1.0
            counts["inhibitory"] += 1
        else:
            counts["unsigned"] += 1
    print(f"  excitatory: {counts['excitatory']:,}  inhibitory: {counts['inhibitory']:,}  "
          f"unsigned: {counts['unsigned']:,} ({100*counts['unsigned']/n:.1f}%)")

    reader = pa.ipc.open_file(pa.memory_map(str(CONNECTIONS), "rb"))
    cols = [f.name for f in reader.schema]
    pre_c = _pick(cols, PRE_CANDIDATES, "presynaptic id")
    post_c = _pick(cols, POST_CANDIDATES, "postsynaptic id")
    w_c = _pick(cols, WEIGHT_CANDIDATES, "synapse count")
    print(f"columns: pre={pre_c!r} post={post_c!r} weight={w_c!r}  (of {cols})")

    def index_of(values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(body_ids, values)
        idx[idx >= n] = 0
        return np.where(body_ids[idx] == values, idx, -1)

    rows, cols_, vals = [], [], []
    total = kept = 0
    syn_total = syn_kept = 0
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column(pre_c).to_numpy(zero_copy_only=False).astype(np.int64)
        post = b.column(post_c).to_numpy(zero_copy_only=False).astype(np.int64)
        w = b.column(w_c).to_numpy(zero_copy_only=False).astype(np.float32)
        total += len(w)
        syn_total += int(w.sum())
        pi, qi = index_of(pre), index_of(post)
        m = (pi >= 0) & (qi >= 0) & (w >= args.min_weight)
        if not m.any():
            continue
        rows.append(pi[m].astype(np.int32))
        cols_.append(qi[m].astype(np.int32))
        vals.append(w[m])
        kept += int(m.sum())
        syn_kept += int(w[m].sum())

    rows = np.concatenate(rows)
    cols_ = np.concatenate(cols_)
    vals = np.concatenate(vals)
    print(f"\nedge rows: {total:,} raw -> {kept:,} annotated-to-annotated "
          f"({100*kept/total:.1f}%)")
    print(f"synapses:  {syn_total:,} raw -> {syn_kept:,} kept")

    signed = vals * sign[rows]
    adj = sp.coo_matrix((signed, (rows, cols_)), shape=(n, n), dtype=np.float32).tocsr()
    adj.sum_duplicates()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flywire_adj_minw{args.min_weight}.npz"
    sp.save_npz(out, adj)
    np.save(OUT_DIR / "flywire_node_ids.npy", body_ids)
    print(f"\nmatrix: {adj.shape[0]:,} x {adj.shape[1]:,}, {adj.nnz:,} stored entries")
    print(f"  excitatory {int((adj.data > 0).sum()):,}  inhibitory {int((adj.data < 0).sum()):,}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
