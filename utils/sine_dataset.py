"""Generate unconditional sine samples directly at the requested sequence length."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset_utils.normalization import make_normalizer, resolve_normalization


class SineDataset(Dataset):
    """Independent sine samples in BLC layout, compatible with PaCoDi evaluation.

    Raw values are in [0, 1]. Minmax normalization uses the generator's known
    range, mapping them to [-1, 1] exactly as in the original SineDataset.
    """

    def __init__(
        self,
        window=128,
        num=30000,
        dim=12,
        save2npy=False,
        neg_one_to_one=True,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        normalization=None,
    ):
        super().__init__()
        if period not in {"train", "test"}:
            raise ValueError(f"period must be train or test, got {period!r}.")
        self.window = int(window)
        self.var_num = int(dim)
        self.sample_num = int(num)
        if min(self.window, self.var_num, self.sample_num) <= 0:
            raise ValueError("window, num, and dim must be positive.")

        self.period = period
        self.save2npy = bool(save2npy)
        self.seed = int(seed)
        if normalization is None:
            normalization = "minmax" if neg_one_to_one else "none"
        self.normalization = resolve_normalization(normalization)
        self.auto_norm = self.normalization == "minmax"
        self.normalize_data = self.normalization != "none"
        self.dir = os.path.join(output_dir, "samples")
        self.rawdata = self.sine_data_generation(
            no=self.sample_num,
            seq_len=self.window,
            dim=self.var_num,
            save2npy=self.save2npy,
            seed=self.seed,
            dir=self.dir,
            period=self.period,
        )
        self.scaler = None
        if self.normalization == "zscore":
            self.scaler = make_normalizer(self.normalization)
            self.scaler.fit(self.rawdata.reshape(-1, self.var_num))
        self.samples = self.normalize(self.rawdata)

    @staticmethod
    def _as_numpy(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype=np.float32)

    def normalize(self, rawdata):
        data = self._as_numpy(rawdata)
        if self.normalization == "minmax":
            return data * 2.0 - 1.0
        if self.scaler is not None:
            return self.scaler.transform(data.reshape(-1, self.var_num)).reshape(data.shape).astype(np.float32)
        return data.copy()

    def unnormalize(self, data):
        data = self._as_numpy(data)
        if self.normalization == "minmax":
            return (data + 1.0) * 0.5
        if self.scaler is not None:
            return self.scaler.inverse_transform(data.reshape(-1, self.var_num)).reshape(data.shape).astype(np.float32)
        return data

    def to_metric_space(self, data):
        """Map minmax model output back to [0, 1], without clipping samples."""
        data = self._as_numpy(data)
        return (data + 1.0) * 0.5 if self.normalization == "minmax" else data

    @staticmethod
    def sine_data_generation(no, seq_len, dim, save2npy=True, seed=123, dir="./", period="train"):
        """Preserve the original interleaved frequency/phase draws and formula."""
        no, seq_len, dim = int(no), int(seq_len), int(dim)
        if min(no, seq_len, dim) <= 0:
            raise ValueError("no, seq_len, and dim must be positive.")
        # A local RandomState preserves the original MT19937 sequence without
        # changing the training process's global NumPy random state.
        draws = np.random.RandomState(seed).uniform(0.0, 0.1, size=(no, dim, 2))
        steps = np.arange(seq_len, dtype=np.float64)
        data = np.empty((no, seq_len, dim), dtype=np.float32)
        # Bound temporary allocations when generating long benchmark windows.
        for start in range(0, no, 256):
            chunk = draws[start:start + 256]
            values = np.sin(chunk[:, :, 0, None] * steps + chunk[:, :, 1, None])
            data[start:start + len(chunk)] = ((values + 1.0) * 0.5).transpose(0, 2, 1)

        if save2npy:
            os.makedirs(dir, exist_ok=True)
            np.save(os.path.join(dir, f"sine_ground_truth_{seq_len}_{period}.npy"), data)
        return data

    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]).float()

    def __len__(self):
        return self.sample_num
