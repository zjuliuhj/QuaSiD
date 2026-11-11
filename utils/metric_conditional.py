from utils.metric_common import (
    align_pairwise_time_series,
    compute_distribution_metrics,
    compute_pairwise_error_metrics,
)


def compute_conditional_metrics(real_data, fake_data):
    """Compute paired conditional-generation metrics for aligned `(N, L, C)` series."""
    real_data, fake_data = align_pairwise_time_series(real_data, fake_data)
    return {
        **compute_pairwise_error_metrics(real_data, fake_data),
        **compute_distribution_metrics(real_data, fake_data),
    }
