"""Resolves the granularity mismatch: dimorphism is per cell-TYPE, fru/dsx
is per-NEURON with a confidence flag. See docs/project_plan.md section (e)
risk #2 and the Phase-0 gate in section (d).

This is the Phase-0 kill-gate's core logic — implemented once we've
inspected the real MaleCNS/flywire_annotations column names (Phase 0,
step 1).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

AggregationRule = Literal["high_confidence_only", "any_positive", "majority"]


def aggregate_type_fru_dsx(neurons: pd.DataFrame, rule: AggregationRule) -> pd.DataFrame:
    """Collapses per-neuron fru/dsx calls to a per-type fru_dsx_positive flag.

    Run all three rules and compare (see risk mitigation in the plan) before
    trusting the Phase-0 cross-tab built on any single rule.
    """
    raise NotImplementedError("Fill in once real MaleCNS column names are confirmed (Phase 0).")


def build_dimorphism_crosstab(
    type_fru_dsx: pd.DataFrame, dimorphism: pd.DataFrame
) -> pd.DataFrame:
    """Builds the 2x2 [sex-different yes/no] x [carries fru/dsx yes/no] table
    that is the Phase-0 gate criterion.
    """
    raise NotImplementedError("Fill in once real MaleCNS column names are confirmed (Phase 0).")
