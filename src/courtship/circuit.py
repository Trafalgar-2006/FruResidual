"""The male courtship song pathway, as neuron groups.

Pathway (Kohatsu & Yamamoto; von Philipsborn et al.; Ding et al.):

    cVA pheromone -> ORN_DA1 -> (projection neurons) -> pC1/P1 command
                  -> pIP10 (descending) -> TN1 / dPR1 (VNC premotor)
                  -> wing motor neurons (T2) -> song

Verified present in the built MaleCNS matrix, with the expected polarity
(2026-10-13): pC1->pIP10 1,941 synapses over 167 pairs; pIP10->TN1 3,475
over 53; TN1->wing motor 6,607 over 460; and ZERO synapses in either
reverse direction tested. pIP10's strongest single target is dPR1, a
song-pathway neuron that was not searched for -- the matrix recovered it
unprompted.

ORN_DA1 -> pC1 is deliberately NOT expected to be direct: that route runs
through antennal-lobe projection neurons, and measures 0 direct synapses,
which is correct rather than a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Groups are matched on the `type` column by regex (anchored at the start).
GROUPS: dict[str, str] = {
    # Sensory input: cVA-responsive olfactory receptor neurons.
    "ORN_DA1": r"ORN_DA1",
    "ORN_VA1v": r"ORN_VA1v",
    # Antennal-lobe projection neurons carrying DA1/VA1v onward.
    "PN_DA1": r"DA1_(lPN|vPN|adPN)",
    "PN_VA1v": r"VA1v_(adPN|vPN)",
    # The courtship command centre.
    "pC1": r"pC1",
    # The descending neuron that drives song.
    "pIP10": r"pIP10",
    # VNC song premotor.
    "TN1": r"TN1",
    "dPR1": r"dPR1",
}

# Wing motor neurons are identified by class + neuromere rather than name:
# the mesothoracic (T2) segment drives the wing muscles.
WING_MOTOR_CLASSES = ("vnc_motor", "vnc_efferent")
WING_MOTOR_NEUROMERE = r"T2|MesoN"

# The readout. Song = wing motor activity, so this is what "is he singing?"
# actually measures.
READOUT = "wing_motor"


# FlyWire's annotation table uses different column names and a coarser
# super_class vocabulary than MaleCNS. Mapped here so one circuit
# definition serves both datasets -- two copies would be free to drift,
# and the whole male/female comparison rests on "the same group" meaning
# the same thing on both sides.
FLYWIRE_COLUMNS = {"bodyId": "root_id", "type": "cell_type", "superclass": "super_class"}
FLYWIRE_DESCENDING = ("descending",)
MALECNS_DESCENDING = ("descending_neuron",)


def normalize_annotations(ann: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with MaleCNS-style column names, whichever dataset
    it came from. Leaves a MaleCNS frame untouched."""
    if "bodyId" in ann.columns:
        return ann
    missing = [v for v in FLYWIRE_COLUMNS.values() if v not in ann.columns]
    if missing:
        raise ValueError(f"unrecognised annotation table; missing {missing}")
    out = ann.rename(columns={v: k for k, v in FLYWIRE_COLUMNS.items()}).copy()
    if "somaNeuromere" not in out.columns:
        out["somaNeuromere"] = None  # brain-only: no neuromeres to filter on
    return out


def descending_ids(ann: pd.DataFrame) -> np.ndarray:
    """Descending neurons, the one pathway stage both datasets contain.

    FlyWire is brain-only, so pIP10, TN1, dPR1 and the wing motor neurons
    are all absent; descending neurons are present in both (1,314 male vs
    1,303 female) and are the brain's output channel. That makes them the
    only honest shared readout for a male/female comparison.
    """
    a = normalize_annotations(ann)
    classes = set(FLYWIRE_DESCENDING) | set(MALECNS_DESCENDING)
    return a.loc[a["superclass"].isin(classes), "bodyId"].to_numpy()


def resolve_groups(ann: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map each named group to the bodyIds matching it in `ann`.

    Returns only non-empty groups, so a dataset missing part of the
    pathway (e.g. a brain-only connectome, which has no VNC) yields fewer
    keys rather than silently-empty ones.
    """
    ann = normalize_annotations(ann)
    out: dict[str, np.ndarray] = {}
    types = ann["type"].astype(str)
    for name, pat in GROUPS.items():
        hit = ann.loc[types.str.match(pat, na=False), "bodyId"].to_numpy()
        if len(hit):
            out[name] = hit

    desc = descending_ids(ann)
    if len(desc):
        out["descending"] = desc

    motor = ann[ann["superclass"].isin(WING_MOTOR_CLASSES)]
    if "somaNeuromere" in motor.columns:
        motor = motor[motor["somaNeuromere"].astype(str).str.contains(
            WING_MOTOR_NEUROMERE, na=False)]
    if len(motor):
        out[READOUT] = motor["bodyId"].to_numpy()
    return out


def to_indices(groups: dict[str, np.ndarray], node_ids: np.ndarray) -> dict[str, np.ndarray]:
    """bodyIds -> row/column positions in the adjacency matrix."""
    pos = {b: i for i, b in enumerate(node_ids)}
    return {
        name: np.array([pos[b] for b in ids if b in pos], dtype=np.int64)
        for name, ids in groups.items()
    }
