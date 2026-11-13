import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ema import EMA
from engine.solver import Trainer


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


def test_trainer_checkpoint_preserves_only_step_and_model(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.train_model = torch.nn.Linear(1, 1, bias=False)
    trainer.ema = EMA(trainer.train_model)
    trainer.device = torch.device("cpu")
    trainer.ckpt_folder = tmp_path
    trainer.logger = None
    trainer.is_main_process = True
    trainer.step = 123
    _set_parameter(trainer.train_model, 4.0)
    trainer.save(2)

    checkpoint = torch.load(tmp_path / "ckpt-2.pt", map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"step", "model"}
    assert checkpoint["step"] == 123
    _set_parameter(trainer.train_model, 0.0)
    trainer.step = 0
    trainer.load(2)
    assert trainer.step == 123
    torch.testing.assert_close(trainer.train_model.weight, torch.tensor([[4.0]]))
    torch.testing.assert_close(trainer.ema.ema_model.weight, trainer.train_model.weight)
