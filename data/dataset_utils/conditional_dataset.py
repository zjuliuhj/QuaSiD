import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset_utils.normalization import make_normalizer, resolve_normalization


class ConditionalDataset(Dataset):
    """Load conditional TS pairs and select train/test by `train_test_prop`.

    Each item is `(x: Lx1, condition: D)`. The split is deterministic under
    `seed`; normalization is fitted on the train split and applied to both splits.
    """

    def __init__(
        self,
        name="Agriculture",
        data_root="./data/MMD",
        window=24,
        train_test_prop=0.9,
        seed=123,
        period="train",
        max_length=32,
        normalization="minmax",
        **kwargs,
    ):
        super().__init__()
        assert period in ["train", "test"], "period must be train or test."
        self.window = window
        self.period = period
        self.name = name
        self.max_length = max_length
        self.normalization = resolve_normalization(normalization)
        self.normalize_data = self.normalization != "none"
        if kwargs:
            raise ValueError(f"Unsupported ConditionalDataset params: {sorted(kwargs)}")
        self.train_test_prop = float(train_test_prop)
        if not 0.0 < self.train_test_prop < 1.0:
            raise ValueError(f"train_test_prop must be in (0, 1), got {train_test_prop}.")

        csv_data = pd.read_csv(os.path.join(data_root, f"{name}.csv"))
        raw_data, condition_text, condition = self.read_data(name, csv_data)
        self.len = raw_data.shape[1]
        self.var_num = raw_data.shape[-1]

        train_raw, test_raw, train_text, test_text, train_condition, test_condition = self.divide(
            raw_data,
            condition_text,
            condition,
            self.train_test_prop,
            seed,
        )

        self.scaler = make_normalizer(self.normalization)
        if self.scaler is not None:
            self.scaler.fit(train_raw.reshape(-1, self.var_num))
            train_data = self._normalize(train_raw)
            test_data = self._normalize(test_raw)
        else:
            train_data = train_raw.copy()
            test_data = test_raw.copy()

        self.samples = train_data if period == "train" else test_data
        self.condition_text = train_text if period == "train" else test_text
        self.condition = train_condition if period == "train" else test_condition
        self.data = self.samples

        self.global_mean = np.mean(train_data, axis=(0, 1)).astype(np.float32)
        self.global_std = (np.std(train_data, axis=(0, 1)) + 1e-6).astype(np.float32)
        self.sample_num = self.samples.shape[0]
        if self.sample_num == 0:
            raise ValueError(f"No {period} samples were produced; check train_test_prop and input rows.")

    def _normalize(self, data):
        """Apply the selected train-fitted scaler to `(N, L, C)` arrays."""
        if len(data) == 0 or self.scaler is None:
            return data.copy()
        shape = data.shape
        return self.scaler.transform(data.reshape(-1, self.var_num)).reshape(shape).astype(np.float32)

    def unnormalize(self, data):
        if not self.normalize_data or self.scaler is None:
            return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        original_shape = data.shape
        data_2d = data.reshape(-1, self.var_num)
        physical_data = self.scaler.inverse_transform(data_2d)
        return physical_data.reshape(original_shape).astype(np.float32)

    def to_metric_space(self, data):
        """Map min-max model values from ``[-1, 1]`` to ``[0, 1]`` for metrics.

        Conditional sampling returns values in the same normalized space used by
        the model. Keep visualizations and physical ``unnormalize`` separate,
        while matching unconditional metric evaluation on the unit interval.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        return (data + 1.0) / 2.0 if self.normalization == "minmax" else data

    @staticmethod
    def divide(data, condition_text, condition, train_test_prop, seed=2023):
        if data.shape[0] != len(condition_text):
            raise ValueError("All inputs must have the same number of rows.")

        rng_state = np.random.get_state()
        np.random.seed(seed)

        size = data.shape[0]
        train_num = int(np.ceil(size * train_test_prop))
        indices = np.random.permutation(size)
        train_idx = indices[:train_num]
        test_idx = indices[train_num:]

        train_data = data[train_idx] if data.ndim == 1 else data[train_idx, :]
        test_data = data[test_idx] if data.ndim == 1 else data[test_idx, :]
        train_text = [condition_text[i] for i in train_idx]
        test_text = [condition_text[i] for i in test_idx]
        train_condition = condition[train_idx, :]
        test_condition = condition[test_idx, :]

        np.random.set_state(rng_state)
        return train_data, test_data, train_text, test_text, train_condition, test_condition

    @staticmethod
    def read_data(name, csv_data):
        condition_text = csv_data["Text"].tolist()
        parsed_series = [ast.literal_eval(item) if isinstance(item, str) else item for item in csv_data["OT"]]
        time_series = np.array(parsed_series, dtype=np.float32)
        time_series = time_series.reshape(*time_series.shape, 1).astype(np.float32)

        if any(
            part in [
                "Agriculture",
                "Climate",
                "Energy",
                "Health",
                "Security",
                "Traffic",
                "Economy",
                "Environment",
                "SocialGood",
                "SUSHI",
            ]
            for part in name.split("_")
        ):
            condition_arrays = csv_data["TextEmbedding"].apply(
                lambda embedding_str: np.array(ast.literal_eval(embedding_str))
            )
        else:
            condition_arrays = csv_data["TextEmbedding"].apply(
                lambda embedding_str: [
                    float(num) for num in embedding_str.replace("[", "").replace("]", "").strip().split()
                ]
            )
        condition = np.array(condition_arrays.tolist(), dtype=np.float32)
        return time_series, condition_text, condition

    def __getitem__(self, index):
        x = self.samples[index]
        condition = self.condition[index]
        return torch.from_numpy(x).float(), torch.from_numpy(condition).float()

    def __len__(self):
        return self.sample_num
