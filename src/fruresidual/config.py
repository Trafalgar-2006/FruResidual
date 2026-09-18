"""Loads configs/config.yaml and .env into a single Config object.

Notebooks and scripts should import this instead of reading paths/tokens
by hand, so the same code runs unchanged on the laptop, Colab, Kaggle, or
the lab machine — only the .env and config.yaml differ per environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @property
    def data_raw(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["data_raw"]

    @property
    def data_interim(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["data_interim"]

    @property
    def data_processed(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["data_processed"]

    @property
    def neuprint_server(self) -> str:
        return self.raw["neuprint"]["server"]

    @property
    def neuprint_dataset(self) -> str:
        return self.raw["neuprint"]["dataset"]

    @property
    def fru_dsx_aggregation(self) -> str:
        return self.raw["fru_dsx_aggregation"]

    @property
    def neuprint_token(self) -> str:
        token = os.environ.get("NEUPRINT_TOKEN")
        if not token:
            raise RuntimeError(
                "NEUPRINT_TOKEN not set. Copy .env.example to .env and fill it in."
            )
        return token

    @property
    def cave_token(self) -> str:
        token = os.environ.get("CAVE_TOKEN")
        if not token:
            raise RuntimeError(
                "CAVE_TOKEN not set. Copy .env.example to .env and fill it in."
            )
        return token


def load_config(config_path: str | Path = PROJECT_ROOT / "configs" / "config.yaml") -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
