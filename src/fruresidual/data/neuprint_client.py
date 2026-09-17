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
