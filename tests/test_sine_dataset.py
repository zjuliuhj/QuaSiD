import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.build_dataloader import build_dataloader
from utils.runtime import load_yaml_config, validate_config
from utils.sine_dataset import SineDataset


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (24, 48, 64, 96, 128, 256, 512)


@pytest.mark.parametrize("length", LENGTHS)
def test_sines_config_loads_direct_generated_windows(length):
    config = copy.deepcopy(load_yaml_config(ROOT / "config/unconditional/sines.yaml"))
    config["model"]["params"]["seq_length"] = length
    config["dataloader"]["dataset"]["params"].update(window=length, num=16)
    config["dataloader"].update(batch_size=4, num_workers=0, shuffle=False)
    validate_config(config, "uncond")
    info = build_dataloader(config, SimpleNamespace(distributed=False, train=True))
    dataset = info["dataset"]
    assert isinstance(dataset, SineDataset)
    assert len(dataset) == 16
    assert dataset.samples.shape == (16, length, config["model"]["params"]["channels"])
    assert dataset.samples.dtype == np.float32
    assert next(iter(info["dataloader"])).shape == (4, length, dataset.var_num)
    np.testing.assert_allclose(dataset.unnormalize(dataset.samples), dataset.rawdata, atol=1e-7)
    np.testing.assert_allclose(dataset.to_metric_space(dataset.samples), dataset.rawdata, atol=1e-7)


def test_generator_matches_original_frequency_phase_draw_order_and_preserves_rng():
    np.random.seed(37)
    before = np.random.get_state()
    actual = SineDataset.sine_data_generation(3, 24, 2, save2npy=False, seed=123)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    rng = np.random.RandomState(123)
    expected = []
    for _ in range(3):
        channels = []
        for _ in range(2):
            freq = rng.uniform(0, 0.1)
            phase = rng.uniform(0, 0.1)
            channels.append([np.sin(freq * step + phase) for step in range(24)])
        expected.append((np.asarray(channels).T + 1.0) * 0.5)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


def test_lengths_keep_the_same_series_prefix():
    short = SineDataset(window=24, num=3, dim=2, seed=123)
    long = SineDataset(window=512, num=3, dim=2, seed=123)
    np.testing.assert_array_equal(short.rawdata, long.rawdata[:, :24])


@pytest.mark.parametrize("normalization", ["minmax", "zscore", "none"])
def test_normalization_round_trip(normalization):
    dataset = SineDataset(window=24, num=8, dim=2, normalization=normalization)
    np.testing.assert_allclose(dataset.unnormalize(torch.from_numpy(dataset.samples)), dataset.rawdata, atol=1e-6)
    assert np.isfinite(dataset.samples).all()


def test_sine_files_are_optional(tmp_path):
    dataset = SineDataset(window=24, num=3, dim=2, output_dir=str(tmp_path))
    assert not (tmp_path / "samples").exists()
    saved = SineDataset(window=24, num=3, dim=2, output_dir=str(tmp_path), save2npy=True)
    np.testing.assert_array_equal(np.load(tmp_path / "samples/sine_ground_truth_24_train.npy"), saved.rawdata)


def test_generated_sines_support_worker_processes():
    dataset = SineDataset(window=64, num=16, dim=5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    assert next(iter(loader)).shape == (4, 64, 5)
