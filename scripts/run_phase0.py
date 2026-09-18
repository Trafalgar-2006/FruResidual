"""Phase 0 kill-gate: is the fru/dsx-unexplained dimorphic-neuron residual
big and structured, or does it collapse to noise? See docs/project_plan.md
section (d), PHASE 0.

Usage:  python scripts/run_phase0.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import (
    build_dimorphism_crosstab,
    load_male_cns_annotations,
    phase0_gate_summary,
)


def main() -> None:
    cfg = load_config()
    df = load_male_cns_annotations(cfg, traced_only=True)

    cfg.data_processed.mkdir(parents=True, exist_ok=True)
    results = {}
    for rule in ("any_positive", "high_confidence_only"):
        ct = build_dimorphism_crosstab(df, rule)
        ct.to_csv(cfg.data_processed / f"phase0_crosstab_{rule}.csv")
        results[rule] = phase0_gate_summary(df, rule)
        print(f"\n=== {rule} ===")
        print(ct)
        print(json.dumps(results[rule], indent=2))

    with open(cfg.data_processed / "phase0_gate_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    both_alive = all(r["verdict"] == "ALIVE" for r in results.values())
    print(f"\n{'='*60}\nGATE DECISION (robust across both aggregation rules): "
          f"{'ALIVE' if both_alive else 'INCONCLUSIVE'}\n{'='*60}")
    return results


if __name__ == "__main__":
    main()
