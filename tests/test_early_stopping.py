"""Validation patience, selected weights, and normal early-stop completion."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from engine.solver import Trainer
from utils.runtime import validate_config

ROOT = Path(__file__).resolve().parents[1]


class TrainLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch):
        return self.weight.square() * batch.mean()


def make_trainer(tmp_path, steps=12, enabled=True, validation=True):
    config = {
        "solver": {
            "train_steps": steps, "save_cycle": 2, "grad_accum_every": 1,
            "validation": {"enabled": validation, "interval": 2, "max_batches": None,
                           "early_stopping": {"enabled": enabled, "patience": 2}},
            "ema": {"decay": 0.995, "update_interval": 1},
            "scheduler": {"target": "engine.lr_sch.CosineAnnealingLRWithWarmup",
                          "params": {"T_max": steps}},
        },
        "dataloader": {"batch_size": 2},
    }
    args = SimpleNamespace(gpu=None, mode="forecast", ckpt_folder=tmp_path, log_interval=1)
    loader = {"dataloader": [torch.ones(2, 1, 1)]}
    return Trainer(config, args, TrainLoss(), loader, val_dataloader=loader)


@pytest.mark.parametrize(
    "losses,steps,expected_step,best_step,early_stopped",
    [([1., 1., 1.], 12, 6, 2, True),
     ([1., 1.1, .9, .9, .95], 12, 10, 6, True),
     ([1., float('nan'), float('inf')], 12, 6, 2, True),
     ([1., .9, .8, .7], 8, 8, 8, False)],
)
def test_training_stops_after_two_misses_and_keeps_best_weights(
    tmp_path, monkeypatch, losses, steps, expected_step, best_step, early_stopped,
):
    trainer = make_trainer(tmp_path, steps)
    values = iter(losses)
    validations = {}

    def validate():
        # The real validation method broadcasts the same scalar to all DDP ranks.
        trainer.last_validation_step = trainer.step
        validations[trainer.step] = trainer.train_model.weight.detach().clone()
        return next(values)

    monkeypatch.setattr(trainer, "validate", validate)
    trainer.train()
    assert trainer.step == expected_step
    assert list(validations) == list(range(2, expected_step + 1, 2))
    assert trainer.early_stopped is early_stopped
    assert trainer.best_validation_step == best_step
    assert trainer.validation_no_improvement == (2 if early_stopped else 0)
    selected = torch.load(tmp_path / "ckpt-best.pt", weights_only=True)
    assert set(selected) == {"step", "model"}
    assert selected["step"] == best_step
    torch.testing.assert_close(selected["model"]["weight"], validations[best_step])
    marker = json.loads((tmp_path / "training-complete.json").read_text())
    assert marker["step"] == expected_step
    if early_stopped:
        assert marker == dict(step=expected_step, reason="early_stopping", train_steps=steps,
                              patience=2, validation_interval=2, validation_max_batches=None)
        # The selected weights remain loadable for subsequent evaluation.
        trainer.load("best")
        torch.testing.assert_close(trainer.ema.ema_model.weight, validations[best_step])
    else:
        assert marker == {"step": steps}


def test_disabling_early_stopping_runs_full_budget(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, steps=8, enabled=False)
    calls = []

    def validate():
        calls.append(trainer.step)
        trainer.last_validation_step = trainer.step
        return 1.0

    monkeypatch.setattr(trainer, "validate", validate)
    trainer.train()
    assert trainer.step == 8
    assert calls == [2, 4, 6, 8]
    assert trainer.early_stopped is False
    assert json.loads((tmp_path / "training-complete.json").read_text()) == {"step": 8}


def test_disabling_validation_also_disables_early_stopping(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, steps=5, validation=False)

    def unexpected_validation():
        pytest.fail("Disabled validation must not run.")

    monkeypatch.setattr(trainer, "validate", unexpected_validation)
    trainer.train()
    assert trainer.step == 5 and not trainer.early_stopping_enabled


def test_final_validation_runs_once_when_off_interval(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, steps=5)
    calls = []

    def validate():
        calls.append(trainer.step)
        trainer.last_validation_step = trainer.step
        return 1.0 / trainer.step

    monkeypatch.setattr(trainer, "validate", validate)
    trainer.train()
    assert calls == [2, 4, 5]
    assert trainer.step == 5 and not trainer.early_stopped


@pytest.mark.parametrize("patience", [0, -1])
def test_invalid_patience_is_rejected(patience):
    config = yaml.safe_load((ROOT / "config/forecasting/etth1.yaml").read_text())
    invalid = copy.deepcopy(config)
    invalid["solver"]["validation"]["early_stopping"]["patience"] = patience
    with pytest.raises(ValueError, match="patience"):
        validate_config(invalid, "forecast")
