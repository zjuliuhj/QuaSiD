import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance

from utils.metric_spectral import compute_spectral_energy_metrics


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
