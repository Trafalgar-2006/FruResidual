"""MaleCNS access via neuPrint (see docs/project_plan.md section b.1)."""

from __future__ import annotations

import pandas as pd
from neuprint import Client, NeuronCriteria, fetch_neurons

from fruresidual.config import Config


def get_client(cfg: Config) -> Client:
    return Client(cfg.neuprint_server, dataset=cfg.neuprint_dataset, token=cfg.neuprint_token)


def fetch_all_neurons(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (neurons_df, roi_counts_df) for every neuron in MaleCNS v1.0.

    neurons_df columns include type/side/superclass/hemilineage/flywireType
    and (per docs/project_plan.md caveat) the exact fru/dsx and neurotransmitter
    column names are unconfirmed until we inspect this DataFrame directly —
    that inspection is the first real step of Phase 0.
    """
    client = get_client(cfg)
    client.fetch_version()
    return fetch_neurons(NeuronCriteria(type=".*"), client=client)


def fetch_roi_info(cfg: Config, body_ids: list[int], batch_size: int = 500) -> pd.DataFrame:
    """Fetches per-neuron ROI innervation (roiInfo) for a specific set of
    bodyIds — targeted, not a full-connectome pull, so this stays fast even
    on a slow connection. Used for the Phase-1 atlas, where we only need
    ROI info for the ~2,368 dimorphic/male-specific neurons, not all 165K.

    Returns one row per bodyId with a raw `roiInfo` JSON string column;
    see fruresidual.features.atlas.parse_roi_info to expand it.
    """
    client = get_client(cfg)
    # Cast to native Python ints explicitly -- numpy int64's repr (e.g. in
    # numpy>=2.0) is "np.int64(123)", not "123", which would corrupt the
    # Cypher list literal below if body_ids came from a pandas column.
    ids = [int(x) for x in body_ids]
    chunks = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        df = client.fetch_custom(
            f"MATCH (n:Neuron) WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, n.roiInfo AS roiInfo"
        )
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)
