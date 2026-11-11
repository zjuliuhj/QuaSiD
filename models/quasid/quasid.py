from dataclasses import dataclass

import torch
import torch.nn as nn

from models.quasid.backbones import build_backbone
from models.quasid.dit_common import InstNormMixin


@dataclass(frozen=True)
class ConditionContext:
    """Condition package with global DiT context and channel-wise spectral FiLM.

    `backbone` stores the global real/imaginary conditioning pair. `channel_mod`
    stores channel-wise spectral FiLM parameters applied before the selected
    backbone processes the diffusion state.
    """

    backbone: object
    channel_mod: object = None


class QuaSiDConditionEncoder(nn.Module):
    """Encode conditions while preserving channel identity for sequence inputs.

    Frequency/time sequence conditions use per-channel features `(B, C, K)` and
    produce both global context `(B or B*C, D)` and channel-wise FiLM. Embedding
    conditions remain global vectors `(B, D_cond)` and do not create channel FiLM.
    """

    def __init__(self, condition_dim, emb_size, condition_domain="frequency", channel_indep=False):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.condition_domain = condition_domain
        self.channel_indep = bool(channel_indep)
        self.proj_r = nn.Linear(self.condition_dim, emb_size)
        self.proj_i = nn.Linear(self.condition_dim, emb_size)
        self.channel_modulation = nn.Sequential(nn.SiLU(), nn.Linear(2 * emb_size, 4, bias=True))
        nn.init.constant_(self.channel_modulation[-1].weight, 0)
        nn.init.constant_(self.channel_modulation[-1].bias, 0)

    def _embedding_context(self, condition):
        """Project embedding condition `(B, D_cond)` into global R/I context."""
        if condition.ndim != 2:
            raise ValueError(f"embedding condition expects shape (B, D), got {tuple(condition.shape)}")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"embedding condition_dim must equal condition width; got {self.condition_dim} and {condition.shape[1]}."
            )
        return ConditionContext((self.proj_r(condition.float()), self.proj_i(condition.float())), None)

    def _channel_features(self, condition):
        """Return per-channel condition features `(R, I): (B, C, K_or_L)`."""
        if condition.ndim != 3:
            raise ValueError(f"{self.condition_domain} condition expects shape (B, L, C), got {tuple(condition.shape)}")
        if self.condition_domain == "frequency":
            spectrum = torch.fft.rfft(condition, dim=1, norm="ortho")
            return spectrum.real.transpose(1, 2).float(), spectrum.imag.transpose(1, 2).float()
        if self.condition_domain == "time":
            features = condition.transpose(1, 2).float()
            return features, features
        raise ValueError(f"Unknown condition_domain={self.condition_domain!r}; use 'time', 'frequency', or 'embedding'.")

    def forward(self, condition):
        """Return `ConditionContext` for a raw condition tensor."""
        if self.condition_domain == "embedding":
            return self._embedding_context(condition)

        cond_r, cond_i = self._channel_features(condition)
        if cond_r.shape[-1] != self.condition_dim:
            raise ValueError(
                f"{self.condition_domain} condition_dim must equal per-channel condition width; "
                f"got condition_dim={self.condition_dim}, condition features={tuple(cond_r.shape)}."
            )

        channel_r = self.proj_r(cond_r)
        channel_i = self.proj_i(cond_i)
        gamma_r, beta_r, gamma_i, beta_i = self.channel_modulation(
            torch.cat([channel_r, channel_i], dim=-1)
        ).unbind(dim=-1)

        if self.channel_indep:
            global_r = channel_r.reshape(-1, channel_r.shape[-1])
            global_i = channel_i.reshape(-1, channel_i.shape[-1])
            channel_mod = tuple(param.reshape(-1, 1, 1) for param in (gamma_r, beta_r, gamma_i, beta_i))
        else:
            global_r = channel_r.mean(dim=1)
            global_i = channel_i.mean(dim=1)
            channel_mod = tuple(param.unsqueeze(1) for param in (gamma_r, beta_r, gamma_i, beta_i))
        return ConditionContext((global_r, global_i), channel_mod)


class QuaSiD(nn.Module, InstNormMixin):
    """Shared 1D frequency-domain QuaSiD interface.

    Owns `x: (B, L, C)`, optional channel-independent reshaping, rFFT branches
    `R/I: (B, F, C_eff)`, condition encoding, spectral covariance statistics,
    noise prediction, Mahalanobis loss, and generation wrappers. Subclasses own
    diffusion-specific noising and reverse sampling.
    """

    sample_desc = "QuaSiD Sampling"
    conditional_desc = "Conditional QuaSiD"
    conditional_default_steps = None

    def __init__(
        self,
        seq_length,
        channels,
        max_step,
        sampling_steps,
        condition_dim=128,
        cfg_scale=0.0,
        backbone="dit_solver_v5",
        use_pmi_corr=True,
        use_loss_weight=False,
        inst_norm=False,
        inst_norm_eps=1e-5,
        freq_keep_ratio=1.0,
        freq_keep_bins=None,
        condition_domain="frequency",
        channel_indep=False,
        emb_size=128,
        patch_size=1,
        num_layers=4,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.channels = channels
        self.max_step = max_step
        self.sampling_steps = int(max_step if sampling_steps is None else sampling_steps)
        self.inst_norm = inst_norm
        self.inst_norm_eps = inst_norm_eps
        self.condition_domain = condition_domain
        self.condition_dim = int(condition_dim)
        self.channel_indep = bool(channel_indep)
        self.cfg_scale = float(cfg_scale)
        self.use_loss_weight = bool(use_loss_weight)
        self.backbone = backbone
        if self.condition_domain == "embedding" and self.channel_indep and self.channels != 1:
            raise ValueError("embedding conditions support channel_indep only when channels=1.")
        self.eff_channels = 1 if self.channel_indep else self.channels

        self.full_spectrum_bins = self.seq_length // 2 + 1
        if freq_keep_bins is None:
            freq_keep_bins = round(self.full_spectrum_bins * float(freq_keep_ratio))
        self.freq_keep_bins = max(1, min(self.full_spectrum_bins, int(freq_keep_bins)))
        self.keep_nyquist = self.seq_length % 2 == 0 and self.freq_keep_bins < self.full_spectrum_bins
        self.modeled_spectrum_bins = self.freq_keep_bins + int(self.keep_nyquist)

        backbone_kwargs = dict(
            channels=self.eff_channels,
            emb_size=emb_size,
            patch_size=patch_size,
            num_layers=num_layers,
            **kwargs,
        )
        backbone_kwargs.update(
            use_pmi_corr=use_pmi_corr,
            spectrum_bins=self.modeled_spectrum_bins,
        )
        self.model = build_backbone(backbone, **backbone_kwargs)
        self.condition_encoder = QuaSiDConditionEncoder(
            self.condition_dim,
            emb_size,
            condition_domain=self.condition_domain,
            channel_indep=self.channel_indep,
        )
        self.to(device)

    def project_condition(self, condition):
        """Encode raw condition into global backbone context and channel-wise FiLM."""
        return self.condition_encoder(condition.to(self.device).float())

    def _reshape_channels_in(self, x):
        """Map `(B, L, C)` to `(B*C, L, 1)` for channel-independent modeling."""
        batch, length, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        return x.transpose(1, 2).reshape(batch * channels, length, 1)

    def _reshape_channels_out(self, x, batch_size, channels):
        """Map `(B*C, L, 1)` back to `(B, L, C)` after channel-independent sampling."""
        return x.reshape(batch_size, channels, x.shape[1], 1).transpose(1, 2).reshape(batch_size, x.shape[1], channels)

    def _prepare_training_batch(self, data):
        """Move a training batch to device and apply optional instance normalization."""
        if isinstance(data, (list, tuple)):
            x, condition = data[0], data[1]
        else:
            x, condition = data, None

        x = x.to(self.device).float()
        if condition is not None:
            condition = condition.to(self.device).float()

        if self.inst_norm:
            if condition is not None and condition.ndim == 3:
                condition, stats = self.inst_normalize(condition)
                x, _ = self.inst_normalize(x, stats=stats)
            else:
                x, _ = self.inst_normalize(x)
        return x, condition

    def _contains_nyquist(self, num_bins):
        """Return whether an equal-length R/I layout stores Nyquist in its final slot."""
        full_grid = num_bins == self.full_spectrum_bins
        cropped_grid = self.keep_nyquist and num_bins == self.modeled_spectrum_bins
        return self.seq_length % 2 == 0 and (full_grid or cropped_grid)

    def _spectral_stats(self, num_bins, device, dtype=torch.float32):
        """Return equal-length R/I covariance and precision for orthonormal rFFT noise."""
        has_nyquist = self._contains_nyquist(num_bins)

        endpoint = torch.zeros(num_bins, dtype=torch.bool, device=device)
        endpoint[0] = True
        if has_nyquist:
            endpoint[-1] = True

        var_r = torch.full((1, num_bins, 1), 0.5, device=device, dtype=dtype)
        var_i = torch.full((1, num_bins, 1), 0.5, device=device, dtype=dtype)
        var_r[:, endpoint, :] = 1.0
        var_i[:, endpoint, :] = 0.0

        inv_r = 1.0 / torch.clamp(var_r, min=1e-7)
        inv_i = torch.zeros_like(var_i)
        inv_i[var_i > 0] = 1.0 / var_i[var_i > 0]
        return {
            "var_r": var_r,
            "var_i": var_i,
            "inv_r": inv_r,
            "inv_i": inv_i,
            "full_bins": self.full_spectrum_bins,
            "modeled_bins": num_bins,
            "low_bins": num_bins if num_bins == self.full_spectrum_bins else num_bins - int(has_nyquist),
            "has_nyquist": has_nyquist,
        }

    def _to_spectrum(self, x):
        """Map `x: (B, L, C_eff)` to cropped rFFT branches plus covariance stats."""
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        if self.keep_nyquist:
            spectrum = torch.cat(
                [spectrum[:, : self.freq_keep_bins, :], spectrum[:, -1:, :]],
                dim=1,
            )
        else:
            spectrum = spectrum[:, : self.freq_keep_bins, :]
        stats = self._spectral_stats(
            spectrum.shape[1],
            spectrum.device,
            spectrum.real.dtype,
        )
        return spectrum.real.float(), spectrum.imag.float(), stats

    def _pad_spectrum(self, x_r, x_i, stats):
        """Restore cropped R/I branches to the full rFFT bin count before irFFT."""
        full_bins = stats["full_bins"]
        if x_r.shape[1] == full_bins:
            return x_r, x_i

        padded_r = torch.zeros(x_r.shape[0], full_bins, x_r.shape[2], device=x_r.device, dtype=x_r.dtype)
        padded_i = torch.zeros_like(padded_r)
        low_bins = stats["low_bins"]
        padded_r[:, :low_bins, :] = x_r[:, :low_bins, :]
        padded_i[:, :low_bins, :] = x_i[:, :low_bins, :]
        if stats["has_nyquist"]:
            padded_r[:, -1:, :] = x_r[:, -1:, :]
        full_stats = self._spectral_stats(full_bins, x_r.device, x_r.dtype)
        return padded_r, padded_i * (full_stats["var_i"] > 0)

    def _to_time(self, x_r, x_i, stats):
        """Map spectral R/I branches back to `x: (B, L, C_eff)` with valid rFFT endpoints."""
        x_r, x_i = self._pad_spectrum(x_r, x_i, stats)
        stats = self._spectral_stats(x_r.shape[1], x_r.device, x_r.dtype)
        spectrum = torch.complex(x_r, x_i * (stats["var_i"] > 0))
        return torch.fft.irfft(spectrum, n=self.seq_length, dim=1, norm="ortho")

    def _spectral_noise_like(self, ref, stats):
        """Sample R/I Gaussian spectral noise using the rFFT covariance convention."""
        noise_r = torch.randn_like(ref) * torch.sqrt(stats["var_r"])
        noise_i = torch.randn_like(ref) * torch.sqrt(stats["var_i"])
        return noise_r, noise_i

    def _initial_spectrum_state(self, batch_size):
        """Create the initial spectral Gaussian state used by reverse samplers."""
        num_bins = self.modeled_spectrum_bins
        stats = self._spectral_stats(num_bins, self.device)
        ref = torch.empty(batch_size, num_bins, self.eff_channels, device=self.device)
        x_r, x_i = self._spectral_noise_like(ref, stats)
        return x_r, x_i, stats

    def _model_timestep(self, t):
        """Convert sampler time to the backbone time embedding domain."""
        return t

    def _apply_condition_modulation(self, x_r, x_i, condition):
        """Apply channel-wise condition FiLM to spectral branches before DiT."""
        if condition is None or condition.channel_mod is None:
            return x_r, x_i
        gamma_r, beta_r, gamma_i, beta_i = condition.channel_mod
        return (1.0 + gamma_r) * x_r + beta_r, (1.0 + gamma_i) * x_i + beta_i

    def _run_backbone(self, x_r, x_i, t, condition, noise_scale, stats):
        """Return epsilon predictions with the configured backbone-owned PMI fusion."""
        return self.model(
            x_r, x_i, t, condition=condition, noise_scale=noise_scale,
            spectral_variance=(stats["var_r"], stats["var_i"]),
        )

    def _predict_noise(self, x_r, x_i, t, stats, noise_scale, condition=None, cfg_scale=0.0):
        """Predict noise with condition-modulated inputs and backbone-owned PMI fusion."""
        t = self._model_timestep(t)
        if cfg_scale > 0 and condition is not None:
            uncond_r, uncond_i = self._run_backbone(
                x_r, x_i, t, None, noise_scale, stats
            )
            model_x_r, model_x_i = self._apply_condition_modulation(x_r, x_i, condition)
            cond_r, cond_i = self._run_backbone(
                model_x_r,
                model_x_i,
                t,
                condition.backbone,
                noise_scale,
                stats,
            )
            pred_r = uncond_r + cfg_scale * (cond_r - uncond_r)
            pred_i = uncond_i + cfg_scale * (cond_i - uncond_i)
        else:
            model_x_r, model_x_i = self._apply_condition_modulation(x_r, x_i, condition)
            backbone_condition = condition.backbone if condition is not None else None
            pred_r, pred_i = self._run_backbone(
                model_x_r,
                model_x_i,
                t,
                backbone_condition,
                noise_scale,
                stats,
            )
        return pred_r, pred_i * (stats["var_i"] > 0)

    def _noise_loss(self, pred_r, pred_i, noise_r, noise_i, stats, weight=None):
        """Mahalanobis noise matching with optional theory weight `lambda_t: (B, 1, 1)`."""
        if weight is None:
            weight = 1.0
        loss_r = (weight * stats["inv_r"] * (pred_r - noise_r) ** 2).mean()
        loss_i = (weight * stats["inv_i"] * (pred_i - noise_i) ** 2).mean()
        return loss_r + loss_i

    def _training_condition(self, cond_emb):
        """Apply classifier-free condition dropout during training."""
        return cond_emb if cond_emb is not None and torch.rand(1).item() >= 0.1 else None

    @torch.no_grad()
    def generate_unconditional(self, batch_size=16):
        """Generate unconditional series; unconditional configs disable instance normalization."""
        if self.inst_norm:
            raise ValueError("Unconditional generation requires inst_norm=false.")
        sample_batch_size = batch_size * self.channels if self.channel_indep else batch_size
        x_r, x_i, stats = self._sample_spectrum(
            batch_size=sample_batch_size,
            num_steps=self.sampling_steps,
            desc=self.sample_desc,
        )
        x = self._to_time(x_r, x_i, stats)
        if self.channel_indep:
            x = self._reshape_channels_out(x, batch_size, self.channels)
        return x

    @torch.no_grad()
    def generate_conditional(
        self,
        batch_size=16,
        cond=None,
        sampling_steps=None,
        cfg_scale=None,
        inst_stats=None,
    ):
        """Generate conditional time series from `cond: (B, ..., ...)` using subclass sampler."""
        if cond is None:
            raise ValueError("generate_conditional requires a condition tensor.")
        if sampling_steps is None:
            sampling_steps = self.conditional_default_steps
        if sampling_steps is None:
            sampling_steps = self.sampling_steps

        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        cond = cond.to(self.device).float()
        if self.inst_norm and cond.ndim == 3:
            cond, inst_stats = self.inst_normalize(cond, stats=inst_stats)
        output_batch_size = cond.shape[0]
        sample_batch_size = output_batch_size
        if self.channel_indep and cond.ndim == 3:
            sample_batch_size = output_batch_size * self.channels
        cond_emb = self.project_condition(cond)
        x_r, x_i, stats = self._sample_spectrum(
            batch_size=sample_batch_size,
            num_steps=sampling_steps,
            condition=cond_emb,
            cfg_scale=cfg_scale,
            desc=self.conditional_desc,
        )
        x = self._to_time(x_r, x_i, stats)
        if self.channel_indep and cond.ndim == 3:
            x = self._reshape_channels_out(x, output_batch_size, self.channels)
        if self.inst_norm:
            x, _ = self.inst_normalize(x, stats=inst_stats, inverse=True)
        return x
