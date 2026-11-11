import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset_utils.normalization import make_normalizer, resolve_normalization


class ForecastingDataset(Dataset):
    """Forecasting windows with time-based train/val/test splits.

    Raw arrays are `(T, C)` or `(N, T, C)`. Each sample returns target
    `(history_length + pred_length, C)` and condition `(history_length, C)`.
    Validation and test windows keep history overlap from the previous split,
    while the predicted future always starts inside the requested period.
    """

    def __init__(
        self,
        data_root,
        history_length,
        pred_length,
        channels,
        stride=1,
        train_ratio=0.7,
        val_ratio=0.1,
        period="train",
        normalization="zscore",
        condition_format="sequence",
        sample_limit=None,
    ):
        super().__init__()
        if period not in {"train", "val", "test"}:
            raise ValueError(f"period must be train, val, or test, got {period!r}.")
        if condition_format != "sequence":
            raise ValueError(
                "ForecastingDataset outputs BLC tensors only; "
                f"condition_format={condition_format!r} is not supported."
            )

        self.data_root = data_root
        self.history_length = int(history_length)
        self.pred_length = int(pred_length)
        self.window = self.history_length + self.pred_length
        self.var_num = int(channels)
        self.stride = int(stride)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.period = period
        self.normalization = resolve_normalization(normalization)
        self.normalize_data = self.normalization != "none"
        self.condition_format = condition_format
        self.sample_limit = None if sample_limit is None else int(sample_limit)
        self._validate_contract()

        rawdata = np.load(data_root).astype(np.float32)
        self.scaler = make_normalizer(self.normalization)
        (
            self.contexts,
            self.targets,
            self.futures,
            self.raw_contexts,
            self.raw_targets,
            self.raw_futures,
        ) = self._prepare(rawdata)
        self.sample_num = self.targets.shape[0]

    def _validate_contract(self):
        """Validate split ratios and window geometry before reading samples."""
        if self.history_length <= 0 or self.pred_length <= 0:
            raise ValueError("history_length and pred_length must be positive.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}.")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                "train_ratio + val_ratio must be smaller than 1 so the test split is non-empty; "
                f"got {self.train_ratio + self.val_ratio}."
            )

    def _split_points(self, length):
        """Return temporal split points `(train_end, val_end)` for a length-`T` series."""
        train_end = int(round(length * self.train_ratio))
        val_end = int(round(length * (self.train_ratio + self.val_ratio)))
        if not 0 < train_end < val_end < length:
            raise ValueError(
                f"Invalid forecasting split for length={length}: "
                f"train_end={train_end}, val_end={val_end}."
            )
        return train_end, val_end

    def _period_source(self, series, train_end, val_end):
        """Select the current period, preserving history lookback at val/test boundaries."""
        if self.period == "train":
            return series[:train_end]
        if self.period == "val":
            return series[max(0, train_end - self.history_length):val_end]
        return series[max(0, val_end - self.history_length):]

    def _normalize(self, data):
        """Apply the selected train-fitted scaler to BLC windows."""
        if self.scaler is None:
            return data.copy()
        shape = data.shape
        return self.scaler.transform(data.reshape(-1, self.var_num)).reshape(shape).astype(np.float32)

    def _sample_indices(self, sample_count):
        if self.sample_limit is None or sample_count <= self.sample_limit:
            return None
        return np.linspace(0, sample_count - 1, self.sample_limit, dtype=np.int64)

    def _make_windows(self, series):
        total = self.history_length + self.pred_length
        if series.shape[0] < total:
            raise ValueError(f"Series length {series.shape[0]} is shorter than forecasting window {total}.")

        contexts, futures = [], []
        for start in range(0, series.shape[0] - total + 1, self.stride):
            mid = start + self.history_length
            end = mid + self.pred_length
            contexts.append(series[start:mid])
            futures.append(series[mid:end])
        return np.asarray(contexts, dtype=np.float32), np.asarray(futures, dtype=np.float32)

    def _prepare_2d(self, rawdata):
        """Build forecasting windows from one multivariate sequence `(T, C)`."""
        if rawdata.shape[1] != self.var_num:
            raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[1]}.")

        train_end, val_end = self._split_points(rawdata.shape[0])
        if self.scaler is not None:
            self.scaler.fit(rawdata[:train_end].reshape(-1, self.var_num))
        raw_contexts, raw_futures = self._make_windows(self._period_source(rawdata, train_end, val_end))
        return raw_contexts, raw_futures

    def _prepare_3d(self, rawdata):
        """Build windows from each sequence in `(N, T, C)` using temporal splits inside every sequence."""
        if rawdata.shape[2] != self.var_num:
            raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[2]}.")

        train_end, val_end = self._split_points(rawdata.shape[1])
        if self.scaler is not None:
            self.scaler.fit(rawdata[:, :train_end, :].reshape(-1, self.var_num))

        contexts, futures = [], []
        for series in rawdata:
            context, future = self._make_windows(self._period_source(series, train_end, val_end))
            contexts.append(context)
            futures.append(future)
        if not contexts:
            raise ValueError("No forecasting windows were produced; check train_ratio, val_ratio, lengths, and sample_limit.")
        return np.concatenate(contexts, axis=0), np.concatenate(futures, axis=0)

    def _prepare(self, rawdata):
        if rawdata.ndim == 2:
            raw_contexts, raw_futures = self._prepare_2d(rawdata)
        elif rawdata.ndim == 3:
            raw_contexts, raw_futures = self._prepare_3d(rawdata)
        else:
            raise ValueError(f"Expected .npy data with shape (T, C) or (N, T, C), got {rawdata.shape}.")

        raw_targets = np.concatenate([raw_contexts, raw_futures], axis=1)

        contexts = self._normalize(raw_contexts) if self.normalize_data else raw_contexts.copy()
        futures = self._normalize(raw_futures) if self.normalize_data else raw_futures.copy()
        targets = self._normalize(raw_targets) if self.normalize_data else raw_targets.copy()

        sample_indices = self._sample_indices(targets.shape[0])
        if sample_indices is not None:
            contexts = contexts[sample_indices]
            futures = futures[sample_indices]
            targets = targets[sample_indices]
            raw_contexts = raw_contexts[sample_indices]
            raw_futures = raw_futures[sample_indices]
            raw_targets = raw_targets[sample_indices]
        return contexts, targets, futures, raw_contexts, raw_targets, raw_futures

    def unnormalize(self, data):
        if not self.normalize_data or self.scaler is None:
            return data
        shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, self.var_num)).reshape(shape).astype(np.float32)

    def __getitem__(self, ind):
        target = torch.from_numpy(self.targets[ind]).float()
        context = self.contexts[ind]
        return target, torch.from_numpy(context).float()

    def __len__(self):
        return self.sample_num
