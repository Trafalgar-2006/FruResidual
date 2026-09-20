"""Find which Codex download actually carries the fru/dsx gene-expression
annotations.

Codex's download page has no file called "gene expression" -- the field
shown as `Gene Expression: Fruitless` in the web UI lives inside one of
the annotation exports, and which one is not documented. Rather than
guess, download the small candidate files into data_raw and run this: it
scans every table for cells mentioning fruitless/doublesex and reports
which file, which column, and how many.

Usage:  python scripts/inspect_codex_download.py [--dir PATH] [--glob PATTERN]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fruresidual.config import load_config

NEEDLES = ("fruitless", "doublesex", "fru", "dsx")
# Files big enough that a full string scan is wasteful, and which cannot
# plausibly hold per-neuron gene annotations anyway.
SKIP_SUBSTRINGS = ("connections", "synapse", "skeleton", "coordinates")


def _read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".feather":
            return pd.read_feather(path)
        sep = "\t" if path.suffix in {".tsv", ".tab"} else ","
        return pd.read_csv(path, sep=sep, low_memory=False, dtype=str)
    except Exception as e:
        print(f"  [could not read: {type(e).__name__}: {e}]")
        return None


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(cfg.data_raw))
    parser.add_argument("--glob", default="*.csv*")
    args = parser.parse_args()

    root = Path(args.dir)
    paths = sorted(p for p in root.glob(args.glob) if p.is_file())
    paths += sorted(root.glob("*.tsv")) + sorted(root.glob("*.feather"))
    paths = [p for p in dict.fromkeys(paths)
             if not any(s in p.name.lower() for s in SKIP_SUBSTRINGS)]

    if not paths:
        raise SystemExit(f"No candidate tables found in {root}")

    print(f"scanning {len(paths)} file(s) in {root}\n")
    hits = []
    for path in paths:
        size_mb = path.stat().st_size / 1e6
        print(f"=== {path.name}  ({size_mb:.1f} MB) ===")
        df = _read_any(path)
        if df is None:
            continue
        print(f"  rows={len(df):,}  columns={list(df.columns)}")

        for col in df.columns:
            s = df[col]
            if s.dtype.kind not in "OU" and str(s.dtype) != "string":
                continue
            low = s.astype("string").str.lower()
            mask = low.notna() & low.str.contains("|".join(NEEDLES), regex=True, na=False)
            n = int(mask.sum())
            if n:
                examples = s[mask].dropna().unique()[:4].tolist()
                print(f"  >>> '{col}': {n:,} rows mention fru/dsx  e.g. {examples}")
                hits.append({"file": path.name, "column": col, "n_rows": n})
        print()

    print("=" * 74)
    if hits:
        print("CANDIDATE gene-expression columns, best first:")
        for h in sorted(hits, key=lambda d: -d["n_rows"]):
            print(f"  {h['n_rows']:>7,}  {h['file']} :: {h['column']}")
        print("\nNext: point the crosscheck at the winning file --")
        print("  python scripts/run_annotation_crosscheck.py --source codex \\")
        print("         --codex-file <that file's name>")
        print("If its column is not auto-detected, add the name to")
        print("_CODEX_COLUMN_CANDIDATES in src/fruresidual/data/flywire_client.py.")
    else:
        print("No fru/dsx mentions found in any scanned file.")
        print("The gene annotations are probably in a file not downloaded yet --")
        print("Cell Types, Classification, and both Community Labels exports are")
        print("the small, plausible candidates on the Codex download page.")


if __name__ == "__main__":
    main()
