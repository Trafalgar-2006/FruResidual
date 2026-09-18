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


def fetch_partner_superclass_profile(cfg: Config, body_ids: list[int], batch_size: int = 200) -> pd.DataFrame:
    """For each bodyId, aggregates connection weight to/from every partner
    superclass, summed over both directions (upstream + downstream) --
    the MaleCNS-side input to the Phase 2 wiring-difference target
    (fruresidual.features.rewiring). Aggregation happens server-side in
    Cypher (sum(weight) grouped by partner superclass), so the result set
    stays small regardless of how many raw edges it's built from -- the
    same targeted-query pattern used for Phase 1's ROI atlas.

    Returns long format: (bodyId, partner_superclass, weight).
    """
    client = get_client(cfg)
    ids = [int(x) for x in body_ids]
    chunks = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        df = client.fetch_custom(
            f"MATCH (n:Neuron)-[c:ConnectsTo]->(m:Neuron) WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, m.superclass AS partner_superclass, sum(c.weight) AS weight "
            "UNION "
            f"MATCH (m:Neuron)-[c:ConnectsTo]->(n:Neuron) WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, m.superclass AS partner_superclass, sum(c.weight) AS weight"
        )
        chunks.append(df)
    long_df = pd.concat(chunks, ignore_index=True)
    return long_df.groupby(["bodyId", "partner_superclass"], as_index=False)["weight"].sum()


def fetch_partner_type_profile(cfg: Config, body_ids: list[int], batch_size: int = 200) -> pd.DataFrame:
    """Like fetch_partner_superclass_profile, but grouped by partner
    TYPE (11,710 possible values) instead of partner superclass (~27
    values) -- the finer granularity Phase 3 needs for "connectivity
    motif" enrichment (does the residual set share specific partner
    identities, not just broad partner regions?). Same server-side
    aggregation pattern, so result size stays proportional to actual
    partner diversity, not raw edge count.

    Returns long format: (bodyId, partner_type, weight).
    """
    client = get_client(cfg)
    ids = [int(x) for x in body_ids]
    chunks = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        df = client.fetch_custom(
            f"MATCH (n:Neuron)-[c:ConnectsTo]->(m:Neuron) WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, m.type AS partner_type, sum(c.weight) AS weight "
            "UNION "
            f"MATCH (m:Neuron)-[c:ConnectsTo]->(n:Neuron) WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, m.type AS partner_type, sum(c.weight) AS weight"
        )
        chunks.append(df)
    long_df = pd.concat(chunks, ignore_index=True)
    return long_df.groupby(["bodyId", "partner_type"], as_index=False)["weight"].sum()


def fetch_population_edges(cfg: Config, body_ids: list[int], batch_size: int = 200) -> pd.DataFrame:
    """Fetches directed edges where BOTH endpoints are in `body_ids` --
    the graph structure for Phase 2's GNN. Restricted to population-
    internal edges (not full 1-hop neighborhoods, which could explode to
    thousands of extra nodes per seed) to keep the graph a size that
    trains full-batch on a 4GB GPU without needing neighbor sampling.

    Returns (source, target, weight).
    """
    client = get_client(cfg)
    ids = [int(x) for x in body_ids]
    chunks = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        df = client.fetch_custom(
            f"MATCH (n:Neuron)-[c:ConnectsTo]->(m:Neuron) WHERE n.bodyId IN {batch} "
            f"AND m.bodyId IN {ids} "
            "RETURN n.bodyId AS source, m.bodyId AS target, c.weight AS weight"
        )
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True).drop_duplicates()
