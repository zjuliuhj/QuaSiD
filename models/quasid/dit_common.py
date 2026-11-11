import math

import torch
import torch.nn as nn


__all__ = [
    "InstNormMixin",
    "TimeEmbedding",
    "get_sinusoidal_positional_embeddings",
    "modulate",
]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0)
    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding.unsqueeze(0)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        return embedding.squeeze(1)


class InstNormMixin:
    """Per-sample z-score normalization over `L` for `x: (B, L, C)`."""

    def inst_normalize(self, x, stats=None, inverse=False):
        """Return transformed `x` and `(mean, std): (B or 1, 1, C)`."""
        if stats is None:
            if inverse:
                raise ValueError("Inverse instance normalization requires statistics.")
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.inst_norm_eps
            ).detach()
            stats = (mean, std)
        else:
            mean, std = (value.to(device=x.device, dtype=x.dtype) for value in stats)
            stats = (mean, std)

        output = x * std + mean if inverse else (x - mean) / std
        return output, stats
