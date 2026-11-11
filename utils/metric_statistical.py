import itertools

import numpy as np


_EPS = np.finfo(np.float64).eps
_QUANTILES = np.linspace(0.01, 0.99, 99)


def _as_time_series(data: np.ndarray) -> np.ndarray:
    """Return float32 time series with shape `(N, L, C)`."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.ndim != 3 or data.shape[0] == 0:
        raise ValueError(f"Expected non-empty time series with shape (N, L, C), got {data.shape}.")
    return data


def _quantile_curve_error(real_values: np.ndarray, fake_values: np.ndarray, relative: bool) -> float:
    real_quantiles = np.quantile(real_values, _QUANTILES)
    fake_quantiles = np.quantile(fake_values, _QUANTILES)
    error = np.abs(fake_quantiles - real_quantiles).mean()
    if not relative:
        return float(error)

    scale = real_quantiles[74] - real_quantiles[24]
    if scale <= _EPS:
        scale = real_quantiles[-1] - real_quantiles[0]
    return float(error / scale) if scale > _EPS else float(error)


def _increment_metrics(real_data: np.ndarray, fake_data: np.ndarray, lags: tuple[int, ...]) -> dict:
    metrics = {}
    for lag in lags:
        if lag >= real_data.shape[1]:
            continue
        channel_errors = []
        for channel in range(real_data.shape[2]):
            real_diff = real_data[:, lag:, channel] - real_data[:, :-lag, channel]
            fake_diff = fake_data[:, lag:, channel] - fake_data[:, :-lag, channel]
            channel_errors.append(_quantile_curve_error(real_diff, fake_diff, relative=True))
        metrics[f"increment_quantile_error_lag_{lag}"] = float(np.mean(channel_errors))

    if not metrics:
        raise ValueError(f"No increment lag fits sequence length {real_data.shape[1]}.")
    metrics["increment_quantile_error_mean"] = float(np.mean(list(metrics.values())))
    return metrics


def _channel_spectral_summary(data: np.ndarray, channel: int) -> tuple[np.ndarray, np.ndarray]:
    series = data[:, :, channel].astype(np.float64, copy=True)
    series -= series.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(series, axis=1, norm="ortho")[:, 1:]) ** 2
    total = power.sum(axis=1, keepdims=True)
    power = np.divide(power, total, out=np.zeros_like(power), where=total > _EPS)

    if power.shape[1] <= 1:
        entropy = np.zeros(len(data), dtype=np.float64)
    else:
        entropy = -(power * np.log(np.maximum(power, _EPS))).sum(axis=1) / np.log(float(power.shape[1]))
    return power.mean(axis=0), entropy


def _spectral_metrics(real_data: np.ndarray, fake_data: np.ndarray) -> dict:
    shape_errors = []
    entropy_errors = []
    for channel in range(real_data.shape[2]):
        real_shape, real_entropy = _channel_spectral_summary(real_data, channel)
        fake_shape, fake_entropy = _channel_spectral_summary(fake_data, channel)
        shape_errors.append(0.5 * np.abs(fake_shape - real_shape).sum())
        entropy_errors.append(_quantile_curve_error(real_entropy, fake_entropy, relative=False))

    return {
        "normalized_psd_total_variation": float(np.mean(shape_errors)),
        "spectral_entropy_quantile_error": float(np.mean(entropy_errors)),
    }


def _ordinal_probabilities(data: np.ndarray, order: int, chunk_size: int = 128) -> np.ndarray:
    patterns = tuple(itertools.permutations(range(order)))
    probabilities = np.zeros((len(patterns), data.shape[2]), dtype=np.float64)
    windows_per_channel = data.shape[0] * (data.shape[1] - order + 1)

    for channel in range(data.shape[2]):
        counts = np.zeros(len(patterns), dtype=np.int64)
        for start in range(0, len(data), chunk_size):
            series = data[start : start + chunk_size, :, channel]
            windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=order, axis=1)
            ranks = np.argsort(windows, axis=-1, kind="stable")
            for index, pattern in enumerate(patterns):
                counts[index] += np.count_nonzero(np.all(ranks == pattern, axis=-1))
        probabilities[:, channel] = counts / windows_per_channel
    return probabilities


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    pattern_count = probabilities.shape[0]
    if pattern_count <= 1:
        return np.zeros(probabilities.shape[1], dtype=np.float64)
    return -(probabilities * np.log(np.maximum(probabilities, _EPS))).sum(axis=0) / np.log(float(pattern_count))


def _ordinal_metrics(real_data: np.ndarray, fake_data: np.ndarray, order: int) -> dict:
    if order < 2 or order > real_data.shape[1]:
        raise ValueError(f"ordinal_order must be in [2, {real_data.shape[1]}], got {order}.")

    real_prob = _ordinal_probabilities(real_data, order)
    fake_prob = _ordinal_probabilities(fake_data, order)
    midpoint = 0.5 * (real_prob + fake_prob)
    real_kl = (real_prob * np.log2(np.maximum(real_prob, _EPS) / np.maximum(midpoint, _EPS))).sum(axis=0)
    fake_kl = (fake_prob * np.log2(np.maximum(fake_prob, _EPS) / np.maximum(midpoint, _EPS))).sum(axis=0)

    return {
        "ordinal_pattern_jsd": float((0.5 * (real_kl + fake_kl)).mean()),
        "permutation_entropy_error": float(np.abs(_entropy(fake_prob) - _entropy(real_prob)).mean()),
    }


def compute_statistical_metrics(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    increment_lags: tuple[int, ...] = (1, 2, 4, 8, 16),
    ordinal_order: int = 3,
) -> dict:
    """Compare temporal statistics of real/fake `(N, L, C)` samples."""
    real_data = _as_time_series(real_data)
    fake_data = _as_time_series(fake_data)
    if real_data.shape[1:] != fake_data.shape[1:]:
        raise ValueError(f"Real/fake temporal shapes differ: {real_data.shape} vs {fake_data.shape}.")

    return {
        **_increment_metrics(real_data, fake_data, increment_lags),
        **_spectral_metrics(real_data, fake_data),
        **_ordinal_metrics(real_data, fake_data, ordinal_order),
    }
