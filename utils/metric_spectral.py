import numpy as np


def _as_time_series(data):
    """Return data as (N, L, C) so rFFT always runs along the temporal axis."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        return data[..., np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Expected time-series data with shape (N, L, C), got {data.shape}.")
    return data


def mean_power_spectrum(data):
    """Compute E[|rFFT(x)|^2] over samples for each frequency bin and channel."""
    data = _as_time_series(data)
    spectrum = np.fft.rfft(data, axis=1, norm="ortho")
    return np.mean(np.abs(spectrum) ** 2, axis=0).astype(np.float64)


def compute_spectral_energy_metrics(real_data, fake_data, high_freq_ratio=0.5):
    """Compare real and generated average spectral energy curves."""
    real_energy = mean_power_spectrum(real_data)
    fake_energy = mean_power_spectrum(fake_data)
    bins = min(real_energy.shape[0], fake_energy.shape[0])
    channels = min(real_energy.shape[1], fake_energy.shape[1])
    real_energy = real_energy[:bins, :channels]
    fake_energy = fake_energy[:bins, :channels]

    # Compare absolute energy preservation and normalized spectral shape.
    energy_error = fake_energy - real_energy
    real_total = np.maximum(real_energy.sum(axis=0, keepdims=True), 1e-12)
    fake_total = np.maximum(fake_energy.sum(axis=0, keepdims=True), 1e-12)
    real_shape = real_energy / real_total
    fake_shape = fake_energy / fake_total
    shape_error = fake_shape - real_shape

    high_start = int(np.floor(bins * float(high_freq_ratio)))
    high_start = min(max(high_start, 0), bins - 1)
    high_shape_error = shape_error[high_start:]

    return {
        "spectral_energy_mse": float(np.mean(energy_error ** 2)),
        "spectral_shape_mse": float(np.mean(shape_error ** 2)),
        "spectral_high_freq_shape_mse": float(np.mean(high_shape_error ** 2)),
    }
