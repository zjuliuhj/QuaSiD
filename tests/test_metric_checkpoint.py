import numpy as np
import pytest
import torch

import utils.metric_context_fid as context_metrics
import utils.metric_unconditional as unconditional_metrics
from utils.metric_discriminative import discriminative_score_metrics
from utils.metric_common import metric_checkpoint_path
from utils.metric_predictive import predictive_score_metrics


@pytest.mark.parametrize("metric, name", [
    ("discriminator", "ds"), ("predictor", "ps"), ("ts2vec", "ts2vec"),
])
def test_checkpoint_path_contains_dataset_horizon_and_settings(metric, name, tmp_path):
    real = np.zeros((20, 8, 3), dtype=np.float32)
    path = metric_checkpoint_path(tmp_path, metric, real, 2000, 128, dataset_name="ETTh1")
    assert path.parent == tmp_path / "etth1" / "horizon_8"
    assert path.name.startswith(f"{name}_steps2000_bs128_")
    assert len(path.stem.rsplit("_", 1)[1]) == 16
    assert path.suffix == ".pt"
    other = metric_checkpoint_path(tmp_path, metric, real, 2000, 128, dataset_name="stocks")
    assert other.parent == tmp_path / "stocks" / "horizon_8"
    assert other != path


def test_checkpoint_path_rejects_dataset_path_traversal(tmp_path):
    real = np.zeros((20, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid metric checkpoint dataset name"):
        metric_checkpoint_path(tmp_path, "discriminator", real, 2000, 128, dataset_name="../other")


@pytest.mark.parametrize("metric", [discriminative_score_metrics, predictive_score_metrics])
def test_checkpoint_is_shared_across_seeds_and_generated_batches(metric, tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    real = rng.random((20, 8, 3), dtype=np.float32)
    fake = rng.random((20, 8, 3), dtype=np.float32)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = metric(real, fake, iterations=2, batch_size=4, device="cpu", checkpoint_dir=tmp_path)

    def unexpected_step(*args, **kwargs):
        pytest.fail("A shared evaluator checkpoint must skip optimizer steps.")

    monkeypatch.setattr(torch.optim.Adam, "step", unexpected_step)
    for seed in range(2022, 2027):
        np.random.seed(seed)
        torch.manual_seed(seed)
        loaded = metric(real, fake, iterations=2, batch_size=4, device="cpu", checkpoint_dir=tmp_path)
        assert loaded == expected
        changed_fake = rng.random(fake.shape, dtype=np.float32)
        metric(real, changed_fake, iterations=2, batch_size=4, device="cpu", checkpoint_dir=tmp_path)
    assert len(list(tmp_path.rglob("*.pt"))) == 1


@pytest.mark.parametrize("metric", [discriminative_score_metrics, predictive_score_metrics])
@pytest.mark.parametrize("changed", ["reference", "horizon", "iterations", "batch_size"])
def test_checkpoint_retrains_for_changed_reference_or_settings(metric, changed, tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    real = rng.random((20, 8, 3), dtype=np.float32)
    fake = rng.random((20, 8, 3), dtype=np.float32)
    metric(real, fake, iterations=2, batch_size=4, device="cpu", checkpoint_dir=tmp_path)

    if changed == "reference":
        real[0, 0, 0] += 0.1
    if changed == "horizon":
        real, fake = real[:, :6], fake[:, :6]
    iterations = 3 if changed == "iterations" else 2
    batch_size = 5 if changed == "batch_size" else 4
    steps = []
    original_step = torch.optim.Adam.step

    def count_step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", count_step)
    metric(
        real, fake, iterations=iterations, batch_size=batch_size,
        device="cpu", checkpoint_dir=tmp_path,
    )
    assert len(steps) == iterations
    assert len(list(tmp_path.rglob("*.pt"))) == 2


def test_ts2vec_is_shared_across_seeds_and_generated_batches(tmp_path, monkeypatch):
    ts2vec = context_metrics.TS2Vec
    original_fit = ts2vec.fit
    fits = []

    def small_model(**kwargs):
        kwargs.update(output_dims=4, hidden_dims=4, depth=1)
        return ts2vec(**kwargs)

    def short_fit(model, train_data, **kwargs):
        fits.append(1)
        return original_fit(model, train_data, n_iters=2)

    monkeypatch.setattr(context_metrics, "TS2Vec", small_model)
    monkeypatch.setattr(ts2vec, "fit", short_fit)
    rng = np.random.default_rng(11)
    real = rng.random((20, 8, 3), dtype=np.float32)
    fake = rng.random((20, 8, 3), dtype=np.float32)
    expected = context_metrics.context_fid(real, fake, device="cpu", checkpoint_dir=tmp_path)
    for seed in range(2022, 2027):
        np.random.seed(seed)
        torch.manual_seed(seed)
        loaded = context_metrics.context_fid(real, fake, device="cpu", checkpoint_dir=tmp_path)
        assert loaded == pytest.approx(expected)
        changed_fake = rng.random(fake.shape, dtype=np.float32)
        context_metrics.context_fid(real, changed_fake, device="cpu", checkpoint_dir=tmp_path)
    assert len(fits) == 1
    assert len(list(tmp_path.rglob("*.pt"))) == 1


def test_learned_metrics_receive_the_same_reference_across_seeds(monkeypatch):
    references = []

    def record_reference(real, fake, **kwargs):
        references.append(real.copy())
        return 0.0

    def record_discriminator(real, fake, **kwargs):
        record_reference(real, fake)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(unconditional_metrics, "discriminative_score_metrics", record_discriminator)
    monkeypatch.setattr(unconditional_metrics, "predictive_score_metrics", record_reference)
    monkeypatch.setattr(unconditional_metrics, "context_fid", record_reference)
    monkeypatch.setattr(unconditional_metrics, "_cross_correlation", lambda *args: 0.0)
    monkeypatch.setattr(unconditional_metrics, "compute_distribution_metrics", lambda *args: {})
    monkeypatch.setattr(unconditional_metrics, "compute_statistical_metrics", lambda *args: {})
    rng = np.random.default_rng(11)
    real = rng.random((40, 8, 3), dtype=np.float32)
    for seed in range(2022, 2027):
        fake = rng.random(real.shape, dtype=np.float32)
        unconditional_metrics.compute_unconditional_metrics(real, fake, max_samples=20, seed=seed)
    assert len(references) == 15
    for reference in references:
        np.testing.assert_array_equal(reference, references[0])
