"""Tests for the surrogate architecture.

Causality is the property worth guarding. A normalisation layer that
reduces over the time axis makes every output depend on the whole
sequence -- including later steps -- and nothing about the loss curve
reveals it. The first version of this model used GroupNorm and leaked
0.033 backwards in time.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_surrogate import Surrogate


@pytest.fixture(scope="module")
def model():
    return Surrogate(16, 5).eval()


def test_output_shape_matches_input(model):
    x = torch.zeros(3, 1, 400)
    assert model(x).shape == (3, 1, 400)


def test_editing_the_last_step_changes_nothing_earlier(model):
    a = torch.randn(1, 1, 400)
    b = a.clone()
    b[0, 0, -1] += 10.0
    delta = (model(a) - model(b)).abs().detach()[0, 0]
    assert float(delta[:-1].max()) == 0.0, "the model sees the future"


def test_a_mid_sequence_edit_affects_only_later_steps(model):
    a = torch.randn(1, 1, 400)
    b = a.clone()
    b[0, 0, 200] += 10.0
    delta = (model(a) - model(b)).abs().detach()[0, 0]
    assert float(delta[:200].max()) == 0.0, "information flowed backwards in time"
    assert float(delta[200:].max()) > 0.0, "the model ignored its input"


def test_receptive_field_covers_the_membrane_time_constant():
    # tau_m is 20 ms; at dt=0.2 ms the field must span well beyond that or
    # the surrogate cannot represent the dynamics it is imitating.
    m = Surrogate(16, 8)
    assert m.receptive_field * 0.2 > 60.0


def test_model_is_small_enough_to_ship_to_a_browser():
    m = Surrogate(40, 8)
    n = sum(p.numel() for p in m.parameters())
    assert n < 100_000, n
