import numpy as np

from utils.metric_common import compute_distribution_metrics


def compute_crps_ensemble(real_data, sample_ensemble):
    real_data = np.asarray(real_data, dtype=np.float32)
    sample_ensemble = np.asarray(sample_ensemble, dtype=np.float32)
    if sample_ensemble.ndim != real_data.ndim + 1:
        raise ValueError(
            f"Expected ensemble shape (R, {real_data.shape}), got {sample_ensemble.shape}."
        )

    absolute_error = np.abs(sample_ensemble - real_data[np.newaxis, ...]).mean()
    run_count = sample_ensemble.shape[0]
    if run_count <= 1:
        ensemble_spread = 0.0
    else:
        sorted_samples = np.sort(sample_ensemble, axis=0)
        weights = 2 * np.arange(1, run_count + 1, dtype=np.float32) - run_count - 1
        weight_shape = (run_count,) + (1,) * (sorted_samples.ndim - 1)
        ensemble_spread = float(
            (2.0 * (sorted_samples * weights.reshape(weight_shape)).sum(axis=0) / (run_count ** 2)).mean()
        )
    return float(absolute_error - 0.5 * ensemble_spread)


def compute_prediction_interval_metrics(real_data, sample_ensemble, coverage):
    lower_q = (1.0 - coverage) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(sample_ensemble, lower_q, axis=0)
    upper = np.quantile(sample_ensemble, upper_q, axis=0)
    real_data = np.asarray(real_data, dtype=np.float32)
    inside = (real_data >= lower) & (real_data <= upper)
    return {
        f"picp_{int(coverage * 100)}": float(inside.mean()),
        f"mpiw_{int(coverage * 100)}": float((upper - lower).mean()),
    }


def compute_probabilistic_forecasting_metrics(real_data, sample_ensemble):
    real_data = np.asarray(real_data, dtype=np.float32)
    sample_ensemble = np.asarray(sample_ensemble, dtype=np.float32)
    if real_data.ndim == 2:
        real_data = real_data[..., np.newaxis]
    if sample_ensemble.ndim == 3:
        sample_ensemble = sample_ensemble[..., np.newaxis]
    if sample_ensemble.ndim != 4:
        raise ValueError(f"Expected forecast ensemble shape (R, N, L, C), got {sample_ensemble.shape}.")
    if sample_ensemble.shape[1:] != real_data.shape:
        raise ValueError(f"Forecast ensemble target shape {sample_ensemble.shape[1:]} != real shape {real_data.shape}.")

    mean_prediction = sample_ensemble.mean(axis=0)
    median_prediction = np.median(sample_ensemble, axis=0)
    mean_error = mean_prediction - real_data
    median_error = median_prediction - real_data
    generated_paths = sample_ensemble.reshape(-1, *real_data.shape[1:])

    metrics = {
        "forecast_runs": int(sample_ensemble.shape[0]),
        "metric_samples": int(real_data.shape[0]),
        "mae_mean": float(np.mean(np.abs(mean_error))),
        "mse_mean": float(np.mean(mean_error ** 2)),
        "rmse_mean": float(np.sqrt(np.mean(mean_error ** 2))),
        "mae_median": float(np.mean(np.abs(median_error))),
        "mse_median": float(np.mean(median_error ** 2)),
        "crps": compute_crps_ensemble(real_data, sample_ensemble),
        **compute_distribution_metrics(real_data, generated_paths),
        "ensemble_std": float(sample_ensemble.std(axis=0).mean()),
    }
    metrics.update(compute_prediction_interval_metrics(real_data, sample_ensemble, 0.5))
    metrics.update(compute_prediction_interval_metrics(real_data, sample_ensemble, 0.9))
    return metrics
