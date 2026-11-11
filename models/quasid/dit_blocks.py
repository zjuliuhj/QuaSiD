import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.quasid.dit_common import modulate


__all__ = ["DiTMLP", "ParallelDiTLayer", "SDPAAttention", "SharedQuadraturePMIBlock"]


class SDPAAttention(nn.Module):
    """Self-attention backed by PyTorch SDPA with optional strict flash mode."""

    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_backend="flash",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_backend = attn_backend
        if self.attn_backend not in {"flash", "sdpa"}:
            raise ValueError("attn_backend must be either 'flash' or 'sdpa'.")
        if self.attn_backend == "flash" and self.head_dim % 8 != 0:
            raise ValueError(
                f"Flash attention expects head_dim to be a multiple of 8, got {self.head_dim}."
            )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _run_attn(self, q, k, v, dropout_p):
        if self.attn_backend == "flash" and q.is_cuda:
            if q.dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError(
                    "Flash attention requires CUDA fp16/bf16 inputs. "
                    "Set solver.mixed_precision to 'bf16' or 'fp16'."
                )
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    is_causal=False,
                )

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
        )

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = self._run_attn(q, k, v, dropout_p)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class DiTMLP(nn.Module):
    """Two-layer DiT feed-forward block with activation and matching dropout."""

    def __init__(self, in_features, hidden_features, out_features, act_layer, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class SharedQuadraturePMIBlock(nn.Module):
    """Shared quadrature encoders and six FiLM vectors from Eq. (22).

    Inputs are (B, K, C). Each K-to-rank encoder is evaluated once and shared
    by both output branches. The six time/context modulation vectors are
    broadcast over feature columns. Decoders remain linear: no output norm
    or activation is applied, preserving affinity in the inputs at fixed context.
    """

    def __init__(self, spectrum_bins, time_dim, corr_rank=None):
        super().__init__()
        self.spectrum_bins = int(spectrum_bins)
        self.time_dim = int(time_dim)
        self.corr_rank = int(time_dim if corr_rank is None else corr_rank)
        if min(self.spectrum_bins, self.time_dim, self.corr_rank) < 1:
            raise ValueError("spectrum_bins, time_dim, and corr_rank must be positive.")

        self.b_r = nn.Linear(self.spectrum_bins, self.corr_rank, bias=False)
        self.b_i = nn.Linear(self.spectrum_bins, self.corr_rank, bias=False)
        self.a_r = nn.Linear(self.corr_rank, self.spectrum_bins, bias=False)
        self.a_i = nn.Linear(self.corr_rank, self.spectrum_bins, bias=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, 6 * self.corr_rank),
        )
        self.zero_init_modulation()

    def zero_init_modulation(self):
        """Start from exactly zero PMI correction in both branches."""
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x_r, x_i, time_embedding, imaginary_mask=None):
        """Return real/imaginary score corrections with shape (B, K, C).

        The caller supplies imaginary_mask=(1, K, 1) for physical DC/Nyquist
        constraints. V5 operates on hidden feature columns, as in stacked Fig. 5;
        this block alone does not identify the full network's score decomposition.
        """
        if x_r.ndim != 3 or x_i.shape != x_r.shape:
            raise ValueError("PMI inputs must have identical (B, K, C) shapes.")
        if x_r.shape[1] != self.spectrum_bins:
            raise ValueError(f"Expected {self.spectrum_bins} frequency bins, got {x_r.shape[1]}.")
        if time_embedding.shape != (x_r.shape[0], self.time_dim):
            raise ValueError("PMI context must have shape (B, time_dim).")

        encoded_r = self.b_r(x_r.transpose(1, 2))
        encoded_i = self.b_i(x_i.transpose(1, 2))
        gamma_rr, gamma_ri, beta_r, gamma_ir, gamma_ii, beta_i = (
            value.unsqueeze(1) for value in self.time_mlp(time_embedding).chunk(6, dim=-1)
        )
        latent_r = gamma_rr * encoded_r + gamma_ri * encoded_i + beta_r
        latent_i = gamma_ir * encoded_r + gamma_ii * encoded_i + beta_i
        delta_r = self.a_r(latent_r).transpose(1, 2)
        delta_i = self.a_i(latent_i).transpose(1, 2)
        if imaginary_mask is not None:
            delta_i = delta_i * imaginary_mask.to(device=delta_i.device, dtype=delta_i.dtype)
        return delta_r, delta_i


class ParallelDiTLayer(nn.Module):
    """Branch-local DiT block for hidden states `R/I: (B, K, D)`."""

    def __init__(self, d_model=128, attn_backend="flash"):
        super().__init__()
        mlp_ratio = 2.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_r = SDPAAttention(d_model, num_heads=4, qkv_bias=True, attn_backend=attn_backend)
        self.mlp_r = DiTMLP(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            out_features=d_model,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_r = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

        self.norm1_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_i = SDPAAttention(d_model, num_heads=4, qkv_bias=True, attn_backend=attn_backend)
        self.mlp_i = DiTMLP(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            out_features=d_model,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_i = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(self, x_r, x_i, c_r, c_i=None):
        """Return branch-local conditional hidden states `(R, I): (B, K, D)`."""
        c_i = c_r if c_i is None else c_i
        residual_r, residual_i = x_r, x_i
        s_msa_r, sc_msa_r, g_msa_r, s_mlp_r, sc_mlp_r, g_mlp_r = self.adaLN_r(c_r).chunk(6, dim=1)
        s_msa_i, sc_msa_i, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i = self.adaLN_i(c_i).chunk(6, dim=1)

        norm_attn_r = modulate(self.norm1_r(x_r), s_msa_r, sc_msa_r)
        norm_attn_i = modulate(self.norm1_i(x_i), s_msa_i, sc_msa_i)
        attn_delta_r = g_msa_r.unsqueeze(1) * self.attn_r(norm_attn_r)
        attn_delta_i = g_msa_i.unsqueeze(1) * self.attn_i(norm_attn_i)

        mft_r = x_r + attn_delta_r
        mft_i = x_i + attn_delta_i
        norm_mlp_r = modulate(self.norm2_r(mft_r), s_mlp_r, sc_mlp_r)
        norm_mlp_i = modulate(self.norm2_i(mft_i), s_mlp_i, sc_mlp_i)
        mlp_delta_r = g_mlp_r.unsqueeze(1) * self.mlp_r(norm_mlp_r)
        mlp_delta_i = g_mlp_i.unsqueeze(1) * self.mlp_i(norm_mlp_i)

        x_cond_r = residual_r + attn_delta_r + mlp_delta_r
        x_cond_i = residual_i + attn_delta_i + mlp_delta_i
        return x_cond_r, x_cond_i
