import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.logger import Logger
from engine.solver import Trainer


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.observed_losses = []

    def forward(self, batch):
        loss = (self.weight - batch).square().mean()
        self.observed_losses.append(float(loss.detach()))
        return loss


def build_trainer(tmp_path, *, interval=2, tensorboard=True, plateau=False, train=True):
    args = SimpleNamespace(gpu=None, ckpt_folder=str(tmp_path), save_dir=str(tmp_path),
                           tensorboard=tensorboard, train=train)
    scheduler = {
        'target': 'engine.lr_sch.CosineAnnealingLRWithWarmup',
        'params': {'T_max': 5, 'warmup': 0, 'min_lr': 0.0001},
    }
    if plateau:
        scheduler = {
            'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
            'params': {'factor': 0.5, 'patience': 0, 'warmup': 0},
        }
    config = {
        'solver': {'train_steps': 5, 'grad_accum_every': 2, 'log_interval': interval,
                   'save_cycle': 100, 'base_lr': 0.001, 'scheduler': scheduler,
                   'ema': {'decay': 0.9, 'update_interval': 1}},
        'dataloader': {'batch_size': 2},
    }
    logger = Logger(args, config['solver'])
    model = LossModel()
    trainer = Trainer(config, args, model, {'dataloader': [
        torch.tensor([1.0, 2.0]), torch.tensor([-0.5, 0.2]),
    ]}, logger)
    return trainer, logger


def test_training_records_window_means_and_final_partial_window(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.stderr.isatty', lambda: False)
    trainer, logger = build_trainer(tmp_path)
    trainer.train()
    logger.close()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    accumulator = EventAccumulator(logger.log_dir).Reload()
    loss_events = accumulator.Scalars('train/loss')
    assert [event.step for event in loss_events] == [2, 4, 5]
    losses = trainer.train_model.observed_losses
    expected = [sum(losses[:4]) / 4, sum(losses[4:8]) / 4, sum(losses[8:]) / 2]
    assert [event.value for event in loss_events] == pytest.approx(expected)
    assert len(accumulator.Tags()['scalars']) == 9
    assert accumulator.Scalars('train/samples_seen')[-1].value == 20
    assert trainer.elapsed_train_seconds >= accumulator.Scalars('train/elapsed_time_s')[-1].value
    files = list(Path(logger.log_dir).glob('events.out.tfevents.*'))
    assert len(files) == 1
    events = list(EventFileLoader(str(files[0])).Load())
    # One metadata event and one event for all seven metrics at each boundary.
    summaries = [event for event in events if event.HasField('summary')]
    assert len(summaries) == 4


def test_disabled_tensorboard_does_not_fetch_training_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.stderr.isatty', lambda: False)
    trainer, logger = build_trainer(tmp_path, tensorboard=False)
    original_cpu, original_item = torch.Tensor.cpu, torch.Tensor.item

    def guard(original):
        def call(tensor, *args, **kwargs):
            assert not inspect.currentframe().f_back.f_code.co_filename.endswith('engine/solver.py')
            return original(tensor, *args, **kwargs)
        return call

    monkeypatch.setattr(torch.Tensor, 'cpu', guard(original_cpu))
    monkeypatch.setattr(torch.Tensor, 'item', guard(original_item))
    trainer.train()
    logger.close()
    assert not list(tmp_path.rglob('events.out.tfevents.*'))


@pytest.mark.parametrize('plateau', [False, True])
def test_logging_frequency_and_switch_preserve_training_updates(tmp_path, monkeypatch, plateau):
    monkeypatch.setattr('sys.stderr.isatty', lambda: False)
    states = []
    for name, interval, enabled in [('dense', 1, True), ('sparse', 50, True), ('off', 50, False)]:
        trainer, logger = build_trainer(tmp_path / name, interval=interval,
                                       tensorboard=enabled, plateau=plateau)
        trainer.train()
        logger.close()
        assert trainer.sch.last_epoch == 5
        states.append((trainer.train_model.weight.detach().clone(),
                       trainer.opt.param_groups[0]['lr'], copy.deepcopy(trainer.ema.state_dict())))
    for state in states[1:]:
        torch.testing.assert_close(states[0][0], state[0], rtol=0, atol=0)
        assert states[0][1] == state[1]
        torch.testing.assert_close(states[0][2]['ema_model']['weight'],
                                   state[2]['ema_model']['weight'], rtol=0, atol=0)


def test_evaluation_without_metrics_creates_no_event_file(tmp_path):
    _, logger = build_trainer(tmp_path, train=False)
    assert logger.tb_writer is None
    logger.flush()
    logger.close()
    logger.close()
    assert not list(tmp_path.rglob('events.out.tfevents.*'))


def test_writer_uses_configured_buffer_and_preserves_scalar_tags(tmp_path):
    args = SimpleNamespace(save_dir=str(tmp_path), tensorboard=True)
    logger = Logger(args, {'tensorboard_max_queue': 128, 'tensorboard_flush_secs': 30})
    logger.log_scalars({'train/loss': 0.5, 'train/grad_norm': 1.0}, 50)
    assert logger.tb_writer.max_queue == 128
    assert logger.tb_writer.flush_secs == 30
    logger.close()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    reader = EventAccumulator(logger.log_dir).Reload()
    assert reader.Scalars('train/loss')[0].step == 50
    assert reader.Scalars('train/grad_norm')[0].value == 1.0


def test_nonpositive_log_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='log_interval'):
        build_trainer(tmp_path, interval=0)
