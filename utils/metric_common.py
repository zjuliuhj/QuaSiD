import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
import torch

from utils.metric_spectral import compute_spectral_energy_metrics


def metric_checkpoint_path(
    checkpoint_dir: str | Path | None,
    metric: str,
    reference_data,
    iterations: int,
    batch_size: int,
    dataset_name: str = "reference",
) -> Path | None:
    """Share evaluator weights for the same real reference data and training settings."""
    if checkpoint_dir is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_name):
        raise ValueError(f"Invalid metric checkpoint dataset name: {dataset_name!r}")

    settings = [2, metric, iterations, batch_size, torch.__version__]
    digest = hashlib.sha256(json.dumps(settings).encode())
    digest.update(np.asarray([len(reference_data)], dtype=np.int64).tobytes())
    for sequence in reference_data:
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        digest.update(np.asarray(sequence.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(sequence).cast("B"))
    metric_name = {"discriminator": "ds", "predictor": "ps", "ts2vec": "ts2vec"}[metric]
    horizon = reference_data[0].shape[0]
    filename = f"{metric_name}_steps{iterations}_bs{batch_size}_{digest.hexdigest()[:16]}.pt"
    return Path(checkpoint_dir) / dataset_name.lower() / f"horizon_{horizon}" / filename


def load_metric_checkpoint(model: torch.nn.Module, path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False

    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print(f"Loaded {type(model).__name__} checkpoint: {path}", flush=True)
    return True


def save_metric_checkpoint(model: torch.nn.Module, path: Path | None) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(model.state_dict(), temporary_path)
    temporary_path.replace(path)
    print(f"Saved {type(model).__name__} checkpoint: {path}", flush=True)


def _as_time_series(data):
    """Return time-series data as float32 `(N, L, C)` for metric computation."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        return data[..., np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Expected time-series data with shape (N, L, C), got {data.shape}.")
    return data


def align_pairwise_time_series(real_data, fake_data):
    """Align paired real/fake series as `(N, L, C)` with identical temporal shape."""
    real_data = _as_time_series(real_data)
    fake_data = _as_time_series(fake_data)
    sample_count = min(len(real_data), len(fake_data))
    if sample_count <= 0:
        raise ValueError("Pairwise metrics require at least one real/fake sample.")

    real_data = real_data[:sample_count]
    fake_data = fake_data[:sample_count]
    if real_data.shape[1:] != fake_data.shape[1:]:
        raise ValueError(f"Paired metric shapes differ: {real_data.shape} vs {fake_data.shape}.")
    return real_data, fake_data


def paired_subset(real_data, fake_data, max_samples=None, seed=123):
    real_data = _as_time_series(real_data)
    fake_data = _as_time_series(fake_data)
    sample_count = min(len(real_data), len(fake_data))
    if max_samples is not None:
        sample_count = min(sample_count, int(max_samples))

    rng = np.random.default_rng(seed)
    real_idx = rng.permutation(len(real_data))[:sample_count]
    fake_idx = rng.permutation(len(fake_data))[:sample_count]
    return real_data[real_idx], fake_data[fake_idx]


def compute_pairwise_error_metrics(real_data, fake_data):
    """Compute paired point errors with T2S-style per-sample WAPE aggregation."""
    real_data, fake_data = align_pairwise_time_series(real_data, fake_data)
    error = fake_data - real_data
    sample_error = np.sum(np.abs(error), axis=(1, 2))
    sample_denom = np.sum(np.abs(real_data), axis=(1, 2))
    valid = sample_denom != 0
    return {
        "mse": float(np.mean(error ** 2)),
        "wape": float(np.mean(sample_error[valid] / sample_denom[valid])) if np.any(valid) else float("nan"),
        "metric_samples": int(len(real_data)),
    }


def compute_wasserstein_distance(real_data, fake_data):
    """Average per-channel Wasserstein distance after flattening `(N, L, C)` samples."""
    real_data = _as_time_series(real_data)
    fake_data = _as_time_series(fake_data)

    channels = min(real_data.shape[-1], fake_data.shape[-1])
    real_flat = real_data[..., :channels].reshape(-1, channels)
    fake_flat = fake_data[..., :channels].reshape(-1, channels)
    distances = [
        scipy_wasserstein_distance(real_flat[:, channel], fake_flat[:, channel])
        for channel in range(channels)
    ]
    return float(np.mean(distances))


def compute_distribution_metrics(real_data, fake_data):
    """Compute distribution-level Wasserstein and spectral metrics for `(N, L, C)` series."""
    return {
        "wasserstein_distance": compute_wasserstein_distance(real_data, fake_data),
        **compute_spectral_energy_metrics(real_data, fake_data),
    }
