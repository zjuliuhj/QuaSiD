import numpy as np

from data.dataset_utils.conditional_dataset import ConditionalDataset
from utils.metric_common import compute_pairwise_error_metrics


def test_conditional_wape_matches_per_sample_t2s_aggregation():
    real = np.array([[[1.0], [1.0]], [[10.0], [10.0]]], dtype=np.float32)
    fake = real + np.array([[[1.0], [1.0]], [[5.0], [5.0]]], dtype=np.float32)

    metrics = compute_pairwise_error_metrics(real, fake)

    assert metrics["mse"] == 13.0
    assert metrics["wape"] == 0.75


def test_conditional_minmax_metric_space_is_unit_interval():
    dataset = ConditionalDataset.__new__(ConditionalDataset)
    dataset.normalization = "minmax"

    converted = dataset.to_metric_space(np.array([[[-1.0], [0.0], [1.0]]], dtype=np.float32))

    np.testing.assert_allclose(converted, np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
