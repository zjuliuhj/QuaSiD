import numpy as np
import torch

from utils.metric_context_fid import context_fid
from utils.metric_cross_correlation import CrossCorrelationLoss
from utils.metric_discriminative import discriminative_score_metrics
from utils.metric_common import compute_distribution_metrics, paired_subset
from utils.metric_predictive import predictive_score_metrics
from utils.metric_statistical import compute_statistical_metrics


def _cross_correlation(real_data, fake_data, seed):
    """Compute cross-correlation on a deterministic subset of `(N, L, C)` samples."""
    x_real = torch.from_numpy(real_data).float()
    x_fake = torch.from_numpy(fake_data).float()
    idx_r = np.random.default_rng(seed + 1).permutation(len(x_real))[: min(1000, len(x_real))]
    idx_f = np.random.default_rng(seed + 2).permutation(len(x_fake))[: min(1000, len(x_fake))]
    corr = CrossCorrelationLoss(x_real[idx_r], name="CrossCorrelationLoss")
    return float(corr.compute(x_fake[idx_f]).item())


def compute_unconditional_metrics(
    real_data,
    fake_data,
    max_samples=3000,
    discriminative_iterations=2000,
    predictive_iterations=5000,
    seed=123,
    device=None,
    dataset_name="reference",
):
    """Compute distribution, learned, and statistical metrics for unconditional samples."""
    # A fixed real subset lets all generation seeds and models share the evaluators.
    real_data, fake_data = paired_subset(real_data, fake_data, max_samples=max_samples, seed=123)
    discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(
        real_data,
        fake_data,
        iterations=discriminative_iterations,
        device=device,
        dataset_name=dataset_name,
    )
    predictive_score = predictive_score_metrics(
        real_data,
        fake_data,
        iterations=predictive_iterations,
        device=device,
        dataset_name=dataset_name,
    )
    context_score = context_fid(real_data, fake_data, device=device, dataset_name=dataset_name)

    return {
        "context_fid": float(context_score),
        "cross_correlation": _cross_correlation(real_data, fake_data, seed),
        **compute_distribution_metrics(real_data, fake_data),
        **compute_statistical_metrics(real_data, fake_data),
        "discriminative_score": float(discriminative_score),
        "discriminative_fake_accuracy": float(fake_accuracy),
        "discriminative_real_accuracy": float(real_accuracy),
        "predictive_score": float(predictive_score),
        "metric_samples": int(len(real_data)),
    }
