from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.quasid.dit_common import (
    TimeEmbedding,
    get_sinusoidal_positional_embeddings,
)
from models.quasid.dit_blocks import (
    ParallelDiTLayer,
    SharedQuadraturePMIBlock,
)


class DiTSolverV5Layer(nn.Module):
    """Hidden fusion using the shared six-gate PMI parameterization.

    This retains the existing epsilon-space residual convention. Hidden fusion
    followed by nonlinear layers is an architectural realization, not a proof
    of an exact output-level base-score/PMI decomposition.
    """

    def __init__(
        self,
        d_model: int,
        spectrum_bins: int,
        use_pmi_corr: bool = True,
        corr_rank: Optional[int] = None,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.dit = ParallelDiTLayer(d_model, attn_backend=attn_backend)
        self.pmi = (
            SharedQuadraturePMIBlock(
                spectrum_bins=spectrum_bins,
                time_dim=d_model,
                corr_rank=corr_rank,
            )
            if use_pmi_corr
            else None
        )

    def forward(
        self,
        x_r: torch.Tensor,
        x_i: torch.Tensor,
        c_r: torch.Tensor,
        c_i: torch.Tensor,
        noise_scale: Optional[torch.Tensor],
        spectral_variance: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return H_DiT - sigma_t * Sigma * Delta for both hidden branches."""
        dit_r, dit_i = self.dit(x_r, x_i, c_r, c_i)
        if self.pmi is None:
            return dit_r, dit_i
        if noise_scale is None or spectral_variance is None:
            raise ValueError("V5 PMI fusion requires noise_scale and spectral_variance.")

        var_r, var_i = spectral_variance
        pmi_context = 0.5 * (c_r + c_i)
        score_pmi_r, score_pmi_i = self.pmi(
            x_r, x_i, pmi_context, imaginary_mask=(var_i > 0)
        )
        return (
            dit_r - noise_scale * var_r * score_pmi_r,
            dit_i - noise_scale * var_i * score_pmi_i,
        )


class DiTSolverV5(nn.Module):
    """Independent DiT streams with blockwise or output-level PMI correction."""

    def __init__(
        self,
        channels: int = 5,
        emb_size: int = 128,
        patch_size: int = 1,
        num_layers: int = 4,
        use_pmi_corr: bool = True,
        spectrum_bins: Optional[int] = None,
        corr_rank: Optional[int] = None,
        attn_backend: str = "flash",
        pmi_mode: str = "blockwise",
    ) -> None:
        super().__init__()
        if patch_size != 1:
            raise ValueError("dit_solver_v5 requires patch_size=1 for frequency-wise covariance fusion.")
        if spectrum_bins is None:
            raise ValueError("dit_solver_v5 requires spectrum_bins from the diffusion wrapper.")
        if pmi_mode not in {"blockwise", "independent"}:
            raise ValueError("pmi_mode must be 'blockwise' or 'independent'.")

        self.patch_size = patch_size
        self.emb_size = emb_size
        self.channels = channels
        self.spectrum_bins = int(spectrum_bins)
        self.pmi_mode = pmi_mode

        self.conv_r = nn.Conv1d(channels, emb_size, kernel_size=1)
        self.conv_i = nn.Conv1d(channels, emb_size, kernel_size=1)
        self.pos_embed = None
        self.layers = nn.ModuleList(
            [
                DiTSolverV5Layer(
                    d_model=emb_size,
                    spectrum_bins=self.spectrum_bins,
                    use_pmi_corr=use_pmi_corr and pmi_mode == "blockwise",
                    corr_rank=corr_rank,
                    attn_backend=attn_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_r = nn.LayerNorm(emb_size)
        self.ln_i = nn.LayerNorm(emb_size)
        self.linear_emb_to_patch_r = nn.Linear(emb_size, channels)
        self.linear_emb_to_patch_i = nn.Linear(emb_size, channels)
        self.time_emb = TimeEmbedding(dim=emb_size)
        self.pmi = (
            SharedQuadraturePMIBlock(self.spectrum_bins, emb_size, corr_rank=corr_rank)
            if use_pmi_corr and pmi_mode == "independent"
            else None
        )

        self.initialize_weights()

    def forward(
        self,
        input_r: torch.Tensor,
        input_i: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        noise_scale: Optional[torch.Tensor] = None,
        spectral_variance: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return epsilon predictions for the shared DDPM and SDE wrappers."""
        if self.pmi is not None and (noise_scale is None or spectral_variance is None):
            raise ValueError("Independent PMI fusion requires noise_scale and spectral_variance.")
        batch_size, seq_length, channels = input_r.shape
        x_r = self.conv_r(input_r.permute(0, 2, 1)).permute(0, 2, 1)
        x_i = self.conv_i(input_i.permute(0, 2, 1)).permute(0, 2, 1)

        if self.pos_embed is None or self.pos_embed.size(1) != seq_length:
            self.pos_embed = get_sinusoidal_positional_embeddings(seq_length, self.emb_size).to(x_r.device)
        pos_embed = self.pos_embed.to(device=x_r.device, dtype=x_r.dtype)
        x_r = x_r + pos_embed
        x_i = x_i + pos_embed

        time_embedding = self.time_emb(t)
        if isinstance(condition, (list, tuple)):
            condition_r, condition_i = condition
            c_r = time_embedding + condition_r
            c_i = time_embedding + condition_i
        else:
            c_r = time_embedding + condition if condition is not None else time_embedding
            c_i = c_r

        for layer in self.layers:
            x_r, x_i = layer(
                x_r,
                x_i,
                c_r,
                c_i,
                noise_scale,
                spectral_variance,
            )

        pred_r = self.linear_emb_to_patch_r(self.ln_r(x_r))
        pred_i = self.linear_emb_to_patch_i(self.ln_i(x_i))
        if self.pmi is not None:
            var_r, var_i = spectral_variance
            # The independent PMI path sees spectral inputs, never DiT hidden states.
            score_pmi_r, score_pmi_i = self.pmi(
                input_r, input_i, 0.5 * (c_r + c_i), imaginary_mask=(var_i > 0)
            )
            pred_r = pred_r - noise_scale * var_r * score_pmi_r
            pred_i = pred_i - noise_scale * var_i * score_pmi_i
        return pred_r, pred_i

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        for layer in self.layers:
            nn.init.constant_(layer.dit.adaLN_r[-1].weight, 0)
            nn.init.constant_(layer.dit.adaLN_r[-1].bias, 0)
            nn.init.constant_(layer.dit.adaLN_i[-1].weight, 0)
            nn.init.constant_(layer.dit.adaLN_i[-1].bias, 0)
            if layer.pmi is not None:
                layer.pmi.zero_init_modulation()
        if self.pmi is not None:
            self.pmi.zero_init_modulation()


DiT_Solver_V5 = DiTSolverV5
