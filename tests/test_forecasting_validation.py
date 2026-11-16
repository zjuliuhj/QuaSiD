"""Forecast split boundaries, current-weight selection, and checkpoint evaluation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data.build_dataloader import build_dataloader_forecasting
from data.dataset_utils.forecasting_dataset import ForecastingDataset
from engine.solver import Trainer
from main import parse_args
from utils.ema import EMA
from utils.runtime import materialize_runtime_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("series_count", [None, 2])
@pytest.mark.parametrize("explicit", [False, True])
def test_temporal_splits_scaler_and_future_boundaries(tmp_path, series_count, explicit):
    series = np.arange(48, dtype=np.float32).reshape(24, 2)
    data = series if series_count is None else np.stack([series, series + 100])
    path = tmp_path / "series.npy"
    np.save(path, data)
    common = dict(data_root=path, history_length=4, pred_length=2, channels=2,
                  train_ratio=0.5, val_ratio=0.25, normalization="zscore")
    if explicit:
        common.update(split_lengths=[10, 5, 5], train_ratio=-1, val_ratio=-1)
    bounds = (10, 15, 20) if explicit else (12, 18, 24)
    train, val, test = [ForecastingDataset(period=period, **common)
                        for period in ("train", "val", "test")]
    expected_fit = data[:bounds[0]] if series_count is None else data[:, :bounds[0]]
    np.testing.assert_allclose(test.scaler.mean_, expected_fit.reshape(-1, 2).mean(axis=0))
    for dataset, start, end in [(val, bounds[0], bounds[1]), (test, bounds[1], bounds[2])]:
        count = end - start - 2 + 1
        assert len(dataset) == count * (series_count or 1)
        np.testing.assert_array_equal(dataset.raw_contexts[0], series[start - 4:start])
        np.testing.assert_array_equal(dataset.raw_futures[0], series[start:start + 2])
        np.testing.assert_array_equal(dataset.raw_futures[count - 1], series[end - 2:end])
        target, context = dataset[0]
        assert target.shape == (6, 2) and context.shape == (4, 2)
        np.testing.assert_allclose(dataset.unnormalize(dataset.targets), dataset.raw_targets, atol=1e-5)


@pytest.mark.parametrize("lengths", [[10, 5], [10, 0, 5], [10, 5.5, 5], [True, 5, 5], [20, 10, 10]])
def test_invalid_fixed_splits_are_rejected(tmp_path, lengths):
    path = tmp_path / "series.npy"
    np.save(path, np.zeros((24, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="split|Split"):
        ForecastingDataset(path, 4, 2, 2, split_lengths=lengths)


def forecast_config(path):
    config = yaml.safe_load((ROOT / "config/forecasting/etth1.yaml").read_text())
    config["model"]["params"].update(seq_length=6, channels=2)
    config["dataloader"]["dataset"]["params"].update(
        data_root=str(path), history_length=4, pred_length=2, channels=2, split_lengths=[10, 5, 5])
    config["dataloader"].update(batch_size=3, sample_size=3, num_workers=0,
                                 pin_memory=False, persistent_workers=False, test_sample_limit=2)
    return config


def test_loaders_preserve_validation_tail_and_only_limit_test(tmp_path):
    path = tmp_path / "series.npy"
    np.save(path, np.arange(48, dtype=np.float32).reshape(24, 2))
    config = forecast_config(path)
    infos = [build_dataloader_forecasting(config, period=period)
             for period in ("train", "val", "test")]
    assert [len(info["dataset"]) for info in infos] == [5, 4, 2]
    assert [info["dataloader"].drop_last for info in infos] == [True, False, False]
    assert [len(batch[0]) for batch in infos[1]["dataloader"]] == [3, 1]
    assert isinstance(infos[0]["dataloader"].sampler, torch.utils.data.RandomSampler)
    assert isinstance(infos[1]["dataloader"].sampler, torch.utils.data.SequentialSampler)
    np.testing.assert_array_equal(infos[2]["dataset"].raw_futures[-1], np.arange(48).reshape(24, 2)[18:20])


class CurrentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, batch):
        return self.weight * batch.mean()


def selection_trainer(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.train_model = CurrentLoss()
    trainer.model = trainer.train_model
    trainer.ema = EMA(trainer.train_model)
    with torch.no_grad():
        trainer.ema.ema_model.weight.fill_(100)
    trainer.device = torch.device("cpu")
    trainer.amp_dtype = torch.bfloat16
    trainer.use_amp = False
    trainer.distributed = False
    trainer.is_main_process = True
    trainer.validation_enabled = True
    trainer.validation_max_batches = None
    trainer.val_dataloader = {"dataloader": [torch.tensor([[1.], [3.]]), torch.tensor([[9.]])]}
    trainer.logger = None
    trainer.ckpt_folder = tmp_path
    trainer.step = 7
    trainer.best_validation_loss = float("inf")
    trainer.best_validation_step = 0
    trainer.last_validation_step = None
    return trainer


def test_validation_weights_partial_batches_and_saves_matching_current_weights(tmp_path):
    trainer = selection_trainer(tmp_path)
    assert trainer.maybe_save_best() == pytest.approx(26 / 3)
    assert trainer.train_model.training
    state = torch.load(tmp_path / "ckpt-best.pt", weights_only=True)
    assert set(state) == {"step", "model"}
    assert state["model"]["weight"].item() == 2
    metadata = json.loads((tmp_path / "ckpt-best.json").read_text())
    assert metadata["best_validation_step"] == 7
    assert metadata["best_validation_loss"] == pytest.approx(26 / 3)
    trainer.step = 8
    trainer.maybe_save_best()  # Equal loss does not overwrite the selected checkpoint.
    assert torch.load(tmp_path / "ckpt-best.pt", weights_only=True)["step"] == 7
    with torch.no_grad():
        trainer.train_model.weight.fill_(1)
    trainer.maybe_save_best()
    assert trainer.best_validation_step == 8
    with torch.no_grad():
        trainer.train_model.weight.fill_(99)
    trainer.load("best")
    assert trainer.train_model.weight.item() == trainer.ema.ema_model.weight.item() == 1
    assert trainer.step == 8


def test_validation_cap_disabled_and_nonfinite_selection(tmp_path):
    trainer = selection_trainer(tmp_path)
    trainer.validation_max_batches = 1
    trainer.train_model.eval()
    assert trainer.validate() == 4
    assert not trainer.train_model.training
    trainer.val_dataloader = {"dataloader": [torch.tensor([[float("nan")]])]}
    trainer.maybe_save_best()
    assert not (tmp_path / "ckpt-best.pt").exists()
    trainer.validation_enabled = False
    assert trainer.validate() is None


@pytest.mark.parametrize("wrapper", ["cont", "disc"])
def test_forecast_training_best_checkpoint_reload_and_metrics(tmp_path, wrapper):
    path = tmp_path / "series.npy"
    rng = np.random.default_rng(17)
    np.save(path, rng.normal(size=(24, 2)).astype(np.float32))
    config = forecast_config(path)
    config["model"]["target"] = f"models.quasid.quasid_{wrapper}.QuaSiD_{wrapper}"
    config["model"]["params"].update(emb_size=32, num_layers=1, corr_rank=3, max_step=8,
                                       sampling_steps=2, attn_backend="sdpa", sampler="ode")
    config["solver"].update(train_steps=3, save_cycle=2, mixed_precision="no", grad_accum_every=1)
    config["solver"]["validation"] = dict(enabled=True, interval=2, max_batches=None)
    config["solver"]["scheduler"] = {"target": "engine.lr_sch.CosineAnnealingLRWithWarmup",
        "params": dict(T_max=3, warmup=0, min_lr=0.0001, warmup_lr=0.0001, verbose=False)}
    config_path = tmp_path / "forecast.yaml"
    config_path.write_text(yaml.safe_dump(config))
    command = [sys.executable, "-B", str(ROOT / "main.py"), "--name", "forecast_smoke",
               "--mode", "forecast", "--config_file", str(config_path),
               "--model_name", f"quasid_{wrapper}", "--dataset_name", "toy"]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for flags in [["--train"], ["--milestone", "best", "--num_samples", "3",
                               "--ensemble_chunk_size", "2", "--skip_visualization"]]:
        result = subprocess.run(command + flags, cwd=tmp_path, env=environment,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    experiment = tmp_path / f"experiments/forecast/quasid_{wrapper}/toy/forecast_smoke"
    metadata = json.loads((experiment / "ckpt/ckpt-best.json").read_text())
    assert metadata["best_validation_step"] in (2, 3)
    assert "Validation loss at step 3" in (experiment / "log/logs/log.txt").read_text()
    state = torch.load(experiment / "ckpt/ckpt-best.pt", weights_only=True)
    assert set(state) == {"step", "model"}
    metrics = pd.read_csv(experiment / "metrics.csv")
    assert metrics.iloc[0]["milestone"] == "best"
    assert metrics.iloc[0]["forecast_runs"] == 3
    assert metrics.iloc[0]["metric_samples"] == 2
    assert np.isfinite(metrics.iloc[0]["crps"])


def test_all_forecast_configs_and_validation_options():
    for path in (ROOT / "config/forecasting").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        validate_config(config, "forecast")
        runtime = materialize_runtime_config(config, "forecast")
        assert "condition_dim" in runtime["model"]["params"]
        assert config["solver"]["validation"] == dict(
            enabled=True, interval=2000, max_batches=None,
            early_stopping=dict(enabled=True, patience=2),
        )
    for key, value in [("interval", 0), ("max_batches", -1)]:
        invalid = copy.deepcopy(config)
        invalid["solver"]["validation"][key] = value
        with pytest.raises(ValueError):
            validate_config(invalid, "forecast")


def test_best_cli_and_mixed_checkpoint_summaries(tmp_path):
    args = parse_args(["--name", "toy", "--config_file", "toy.yaml", "--milestone", "best"])
    assert args.milestone == "best"
    assert parse_args(["--name", "toy", "--config_file", "toy.yaml", "--milestone", "2"]).milestone == 2
    spec = importlib.util.spec_from_file_location("summary", ROOT / "scripts/experiments/summarize_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "metrics.csv"
    pd.DataFrame([dict(experiment="toy", milestone=milestone, seed=seed, crps=value)
                  for milestone in (2, "best") for seed, value in [(123, 1.0), (456, 3.0)]]).to_csv(path, index=False)
    for milestone in (2, "best"):
        result = module.summarize_metrics(path, "toy", milestone, [123, 456])
        assert result.iloc[0]["mean"] == 2


def test_forecast_launcher_defaults_and_best_summary(tmp_path):
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    shutil.copytree(ROOT / "config", tmp_path / "config")
    environment = dict(os.environ, DRY_RUN="true", LOG_ROOT=str(tmp_path / "queue"), GPU_IDS="0",
                       PYTHON_BIN=sys.executable)
    result = subprocess.run(["bash", "scripts/experiments/run_forecasting.sh"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "HISTORY=168 HORIZON=192" in result.stdout
    assert "DATASET=national_illness HISTORY=168 HORIZON=36" in result.stdout
    assert "MILESTONE=best" in result.stdout
    assert "--num_samples 20 --ensemble_chunk_size 4 --eval_sample_limit 20000" in result.stdout
    capture = tmp_path / "commands.txt"
    fake_python = tmp_path / "capture-python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CAPTURE"\n')
    fake_python.chmod(0o755)
    environment.update(MODE="forecast", DATASET="etth1", RUN_TRAIN="false", EVAL_SEEDS="1,2",
                       PYTHON_BIN=str(fake_python), CAPTURE=str(capture))
    result = subprocess.run(["bash", "scripts/experiments/train_eval.sh"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    commands = capture.read_text()
    assert commands.count("--milestone best") == 3  # Two evaluations and one summary.
