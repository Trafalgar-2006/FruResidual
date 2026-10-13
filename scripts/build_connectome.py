"""Build the signed, weighted MaleCNS connectome as a sparse matrix.

This is the substrate everything else runs on, so it is deliberately
paranoid about two things that would silently produce a wrong network:

1. The raw weights file is the full segmentation at minconf 0.5. It has
   151.9M edge rows spanning ~13.8M distinct bodies -- overwhelmingly tiny
   unproofread fragments, not neurons. Only 16.8% of rows have BOTH
   endpoints in the 165,122 traced set. Building without that restriction
   gives a graph that is ~99% debris.

2. Sign comes from the PRESYNAPTIC neuron's transmitter, not the edge.
   Getting this backwards (signing by the postsynaptic cell) silently
   inverts a large fraction of the network while leaving every summary
   statistic looking plausible.

Sign convention is the standard one for Drosophila fast transmission:
acetylcholine excitatory; GABA, glutamate (GluCl) and histamine
inhibitory. Monoamines (dopamine/octopamine/serotonin, 541 neurons) are
neuromodulators rather than fast transmitters and are given sign 0 by
default, as are neurons with no confident call (3,602, i.e. 2.2%).
UNKNOWN_SIGN makes that choice explicit and testable rather than buried.

Usage:  python scripts/build_connectome.py [--min-weight N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from fruresidual.config import load_config
from fruresidual.features.atlas import load_neurotransmitters
from fruresidual.features.fru_dsx_join import load_male_cns_annotations

WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
OUT_DIR = Path("C:/CourtshipData/processed")

EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate", "histamine"}
# Monoamines and unresolved calls. 0 = contributes no fast current.
UNKNOWN_SIGN = 0.0


def _index_of(values: np.ndarray, sorted_table: np.ndarray) -> np.ndarray:
    """Position of each value in sorted_table; -1 where absent."""
    idx = np.searchsorted(sorted_table, values)
    idx[idx >= len(sorted_table)] = 0
    hit = sorted_table[idx] == values
    return np.where(hit, idx, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weight", type=int, default=1,
                    help="drop edges with fewer than N synapses (default 1 = keep all)")
    args = ap.parse_args()

    cfg = load_config()
    ann = load_male_cns_annotations(cfg, traced_only=True)
    body_ids = np.sort(ann["bodyId"].to_numpy().astype(np.int64))
    n = len(body_ids)
    print(f"traced neurons (graph nodes): {n:,}")

    # --- sign per neuron, from its own transmitter ---
    nt = load_neurotransmitters(cfg)
    nt_map = dict(zip(nt["bodyId"].to_numpy(), nt["consensus_nt"].to_numpy()))
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

    # --- stream the edge list, keeping traced -> traced only ---
    src = cfg.data_raw / WEIGHTS_FILE
    reader = pa.ipc.open_file(pa.memory_map(str(src), "rb"))
    rows, cols, vals = [], [], []
    total_rows = kept_rows = 0
    syn_total = syn_kept = 0

    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column("body_pre").to_numpy(zero_copy_only=False)
        post = b.column("body_post").to_numpy(zero_copy_only=False)
        w = b.column("weight").to_numpy(zero_copy_only=False)
        total_rows += len(w)
        syn_total += int(w.sum())

        pi = _index_of(pre, body_ids)
        qi = _index_of(post, body_ids)
        m = (pi >= 0) & (qi >= 0) & (w >= args.min_weight)
        if not m.any():
            continue
        rows.append(pi[m].astype(np.int32))
        cols.append(qi[m].astype(np.int32))
        vals.append(w[m].astype(np.float32))
        kept_rows += int(m.sum())
        syn_kept += int(w[m].sum())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    print(f"\nedge rows: {total_rows:,} raw -> {kept_rows:,} traced-to-traced "
          f"({100*kept_rows/total_rows:.1f}%)")
    print(f"synapses:  {syn_total:,} raw -> {syn_kept:,} kept")

    # Sign by the PRESYNAPTIC neuron.
    signed = vals * sign[rows]
    n_zeroed = int((signed == 0).sum())
    print(f"edges carrying no sign (unsigned presynaptic neuron): {n_zeroed:,} "
          f"({100*n_zeroed/len(signed):.1f}%)")

    # rows = presynaptic, cols = postsynaptic. A[i, j] is i -> j.
    adj = sp.coo_matrix((signed, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    adj.sum_duplicates()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"malecns_adj_minw{args.min_weight}.npz"
    sp.save_npz(out, adj)
    np.save(OUT_DIR / "malecns_node_ids.npy", body_ids)

    print(f"\nmatrix: {adj.shape[0]:,} x {adj.shape[1]:,}, {adj.nnz:,} stored entries")
    print(f"  excitatory entries: {int((adj.data > 0).sum()):,}")
    print(f"  inhibitory entries: {int((adj.data < 0).sum()):,}")
    deg_out = np.diff(adj.indptr)
    print(f"  out-degree: mean {deg_out.mean():.1f}, max {deg_out.max():,}, "
          f"isolated (no outputs) {int((deg_out == 0).sum()):,}")
    print(f"\nsaved -> {out}")
    print(f"saved -> {OUT_DIR / 'malecns_node_ids.npy'}")


if __name__ == "__main__":
    main()
