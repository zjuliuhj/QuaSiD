import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ema import EMA


def _set_parameter(model, value):
    with torch.no_grad():
        model.weight.fill_(value)


def test_ema_syncs_on_first_scheduled_update_then_averages():
    model = torch.nn.Linear(1, 1, bias=False)
    _set_parameter(model, 0.0)
    ema = EMA(model, beta=0.5, update_every=2)

    _set_parameter(model, 2.0)
    ema.update()
    assert ema.initted is False
    assert ema.ema_model.weight.item() == 0.0

    ema.update()
    assert ema.initted is True
    assert ema.ema_model.weight.item() == 2.0

    _set_parameter(model, 6.0)
    ema.update()
    ema.update()
    assert ema.ema_model.weight.item() == 4.0


def test_ema_checkpoint_round_trip_preserves_initialized_state():
    model = torch.nn.Linear(1, 1, bias=False)
    ema = EMA(model, update_every=1)
    ema.update()

    restored = EMA(torch.nn.Linear(1, 1, bias=False), update_every=1)
    restored.load_state_dict(ema.state_dict())

    assert restored.initted is True
    assert restored.step == 1
    assert torch.equal(restored.ema_model.weight, ema.ema_model.weight)
