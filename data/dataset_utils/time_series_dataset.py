import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset_utils.normalization import make_normalizer, resolve_normalization


class TimeSeriesDataset(Dataset):
    """Window a `.npy` time series into BLC samples with train-fitted normalization."""

    def __init__(
        self,
        data_root,
        window,
        channels,
        stride=1,
        split_ratio=0.9,
        period="train",
        normalization="minmax",
    ):
        super().__init__()
        assert period in ["train", "test"], "period must be train or test."

        self.data_root = data_root
        self.window = int(window)
        self.var_num = int(channels)
        self.stride = int(stride)
        self.period = period
        self.normalization = resolve_normalization(normalization)
        self.normalize_data = self.normalization != "none"

        rawdata = np.load(data_root).astype(np.float32)
        self.rawdata = self._to_windows(rawdata)
        if self.rawdata.shape[1] != self.window or self.rawdata.shape[2] != self.var_num:
            raise ValueError(
                f"Expected data shape (N, {self.window}, {self.var_num}), got {self.rawdata.shape}."
            )

        train_raw, test_raw = self._split(self.rawdata, split_ratio)
        if len(train_raw) == 0:
            raise ValueError("Train split is empty; check split_ratio, window, and stride.")

        self.scaler = make_normalizer(self.normalization)
        if self.scaler is not None:
            self.scaler.fit(train_raw.reshape(-1, self.var_num))

        if self.normalize_data:
            train_data = self._normalize(train_raw)
            test_data = self._normalize(test_raw)
        else:
            train_data = train_raw.copy()
            test_data = test_raw.copy()

        self.samples = train_data if period == "train" else test_data
        self.sample_num = self.samples.shape[0]
        if self.sample_num == 0:
            raise ValueError(f"No {period} windows were produced; check split_ratio, window, and stride.")

    def _normalize(self, data):
        """Apply the selected train-fitted scaler while preserving empty inactive splits."""
        if len(data) == 0 or self.scaler is None:
            return data.copy()
        shape = data.shape
        return self.scaler.transform(data.reshape(-1, self.var_num)).reshape(shape).astype(np.float32)

    def _to_windows(self, rawdata):
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

        if rawdata.ndim == 3:
            if rawdata.shape[2] != self.var_num:
                raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[2]}.")
            if rawdata.shape[1] < self.window:
                raise ValueError(f"Series length {rawdata.shape[1]} is shorter than window {self.window}.")

            samples = []
            for series in rawdata:
                for start in range(0, series.shape[0] - self.window + 1, self.stride):
                    samples.append(series[start:start + self.window])
            return np.asarray(samples, dtype=np.float32)

        if rawdata.ndim != 2:
            raise ValueError(f"Expected .npy data with shape (T, C) or (N, T, C), got {rawdata.shape}.")
        if rawdata.shape[1] != self.var_num:
            raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[1]}.")

        sample_num = rawdata.shape[0] - self.window + 1
        if sample_num <= 0:
            raise ValueError(f"Data length {rawdata.shape[0]} is shorter than window {self.window}.")

        return np.asarray(
            [rawdata[start:start + self.window] for start in range(0, sample_num, self.stride)],
            dtype=np.float32,
        )

    @staticmethod
    def _split(data, split_ratio):
        split = int(round(len(data) * float(split_ratio)))
        return data[:split], data[split:]

    def unnormalize(self, data):
        if not self.normalize_data or self.scaler is None:
            return data
        shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, self.var_num)).reshape(shape).astype(np.float32)

    def to_metric_space(self, data):
        """Map minmax model values from `[-1, 1]` to `[0, 1]` for metrics."""
        data = np.asarray(data, dtype=np.float32)
        return (data + 1.0) / 2.0 if self.normalization == "minmax" else data

    def __getitem__(self, ind):
        return torch.from_numpy(self.samples[ind]).float()

    def __len__(self):
        return self.sample_num
