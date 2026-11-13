import sys

import torch
from tqdm.auto import tqdm

from models.quasid.quasid import QuaSiD


class QuaSiD_cont(QuaSiD):
    """Continuous-time VP-SDE QuaSiD variant.

    Uses the shared QuaSiD frequency-domain interface and owns only the
    continuous schedule: `t in [time_eps, 1]`, closed-form marginal noising, and
    reverse SDE or probability-flow ODE sampling for R/I spectral branches.
    """

    conditional_default_steps = 50

    def __init__(
        self,
        seq_length,
        channels,
        max_step=1000,
        sampling_steps=250,
        beta_min=0.1,
        beta_max=10.0,
        time_eps=1e-5,
        sampler="sde",
        condition_dim=128,
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
        beta_min = float(beta_min)
        beta_max = float(beta_max)
        time_eps = float(time_eps)
        if not 0.0 < beta_min <= beta_max:
            raise ValueError("VP-SDE beta schedule requires 0 < beta_min <= beta_max.")
        if not 0.0 < time_eps < 1.0:
            raise ValueError(f"time_eps must be in (0, 1), got {time_eps}.")
        if sampler not in {"sde", "ode"}:
            raise ValueError(f"sampler must be 'sde' or 'ode', got {sampler!r}.")
        super().__init__(
            seq_length=seq_length,
            channels=channels,
            max_step=float(max_step),
            sampling_steps=sampling_steps,
            condition_dim=condition_dim,
            backbone=backbone,
            use_pmi_corr=use_pmi_corr,
            use_loss_weight=use_loss_weight,
            inst_norm=inst_norm,
            inst_norm_eps=inst_norm_eps,
            freq_keep_ratio=freq_keep_ratio,
            freq_keep_bins=freq_keep_bins,
            condition_domain=condition_domain,
            channel_indep=channel_indep,
            emb_size=emb_size,
            patch_size=patch_size,
            num_layers=num_layers,
            device=device,
            **kwargs,
        )
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_eps = time_eps
        self.sampler = sampler
        self.sample_desc = f"{sampler.upper()} Sampling"
        self.conditional_desc = f"Conditional {sampler.upper()}"

    def _model_timestep(self, t):
        """Scale continuous SDE time to the backbone embedding range `[0, max_step - 1]`."""
        return t * (self.max_step - 1)

    def get_beta(self, t):
        """Linear VP-SDE diffusion rate beta(t)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_marginal_prob(self, t):
        """Closed-form VP-SDE marginal: mean=sqrt(alpha_bar_t), scale=sqrt(1-alpha_bar_t)."""
        # log_mean = -0.5 * int_0^t beta(s) ds, so mean = sqrt(alpha_bar_t)
        # is the closed-form VP-SDE marginal after integrating the linear beta schedule.
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean)
        noise_scale = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
        return mean[:, None, None], noise_scale[:, None, None]

    def loss_weight(self, t, noise_scale):
        """Return epsilon-space VP-SDE weight beta(t) / (2 sigma(t)^2)."""
        noise_variance = torch.clamp(noise_scale.pow(2), min=1e-12)
        beta_t = self.get_beta(t)[:, None, None]
        return beta_t / (2.0 * noise_variance)

    def _sample_training_time(self, batch_size):
        """Sample continuous VP-SDE times uniformly on [time_eps, 1)."""
        return torch.rand(batch_size, device=self.device) * (1.0 - self.time_eps) + self.time_eps

    def forward(self, data, **kwargs):
        x, condition = self._prepare_training_batch(data)
        cond_emb = self.project_condition(condition) if condition is not None else None
        if self.channel_indep:
            x = self._reshape_channels_in(x)
        x_r, x_i, stats = self._to_spectrum(x)
        t = self._sample_training_time(x_r.shape[0])
        mean, noise_scale = self.get_marginal_prob(t)

        noise_r, noise_i = self._spectral_noise_like(x_r, stats)
        xt_r = x_r * mean + noise_r * noise_scale
        xt_i = x_i * mean + noise_i * noise_scale

        pred_r, pred_i = self._predict_noise(
            xt_r,
            xt_i,
            t,
            stats,
            noise_scale=noise_scale,
            condition=self._training_condition(cond_emb),
        )
        weight = self.loss_weight(t, noise_scale) if self.use_loss_weight else None
        return self._noise_loss(pred_r, pred_i, noise_r, noise_i, stats, weight=weight)

    @torch.no_grad()
    def _euler_maruyama_step(
        self,
        x_r,
        x_i,
        t_value,
        dt,
        stats,
        condition=None,
        cfg_scale=0.0,
        add_noise=True,
    ):
        """One reverse VP-SDE Euler-Maruyama step for spectral R/I branches."""
        batch = x_r.shape[0]
        t = torch.full((batch,), float(t_value), device=self.device)
        _, noise_scale = self.get_marginal_prob(t)
        beta_t = self.get_beta(t)[:, None, None]
        g_t = torch.sqrt(beta_t)

        eps_r, eps_i = self._predict_noise(
            x_r,
            x_i,
            t,
            stats,
            noise_scale=noise_scale,
            condition=condition,
            cfg_scale=cfg_scale,
        )
        noise_scale = torch.clamp(noise_scale, min=1e-4)

        f_r = -0.5 * beta_t * x_r
        f_i = -0.5 * beta_t * x_i
        # Heteroscedastic score-noise identity for the full reverse-SDE drift:
        # Sigma @ score = -eps / noise_scale, so the score term is -g(t)^2 Sigma @ score.
        # Sigma_r/i also determine the spectral Wiener variance through var_r/var_i below.
        sigma_score_r = -eps_r / (noise_scale + 1e-7)
        sigma_score_i = -eps_i / (noise_scale + 1e-7)
        reverse_drift_r = f_r - g_t.pow(2) * sigma_score_r
        reverse_drift_i = f_i - g_t.pow(2) * sigma_score_i

        dt_scale = torch.sqrt(torch.tensor(abs(dt), device=self.device))
        # Keep the random stream of the established SDE sampler across batches.
        dw_r = torch.randn_like(x_r) * dt_scale * torch.sqrt(stats["var_r"])
        dw_i = torch.randn_like(x_i) * dt_scale * torch.sqrt(stats["var_i"])

        next_r = x_r + reverse_drift_r * dt
        next_i = x_i + reverse_drift_i * dt
        if add_noise:
            next_r = next_r + g_t * dw_r
            next_i = next_i + g_t * dw_i
        return next_r, next_i * (stats["var_i"] > 0)

    @torch.no_grad()
    def _ode_step(
        self,
        x_r,
        x_i,
        t_value,
        next_t_value,
        stats,
        condition=None,
        cfg_scale=0.0,
    ):
        """Integrate the probability-flow ODE with a continuous DDIM update."""
        t = torch.full((x_r.shape[0],), float(t_value), device=self.device)
        next_t = torch.full_like(t, float(next_t_value))
        mean, noise_scale = self.get_marginal_prob(t)
        next_mean, next_noise_scale = self.get_marginal_prob(next_t)
        eps_r, eps_i = self._predict_noise(
            x_r,
            x_i,
            t,
            stats,
            noise_scale=noise_scale,
            condition=condition,
            cfg_scale=cfg_scale,
        )
        clean_r = (x_r - noise_scale * eps_r) / mean
        clean_i = (x_i - noise_scale * eps_i) / mean
        next_r = next_mean * clean_r + next_noise_scale * eps_r
        next_i = next_mean * clean_i + next_noise_scale * eps_i
        return next_r, next_i * (stats["var_i"] > 0)

    @torch.no_grad()
    def _sample_spectrum(self, batch_size, num_steps, condition=None, cfg_scale=0.0, desc="SDE Sampling"):
        """Sample R/I spectrum with the selected continuous-time integrator."""
        num_steps = int(num_steps)
        x_r, x_i, stats = self._initial_spectrum_state(batch_size)
        time_steps = torch.linspace(1.0, self.time_eps, num_steps + 1, device=self.device)
        if self.sampler == "ode":
            # The final update predicts x_0 without evaluating the score at zero noise.
            time_steps[-1] = 0.0
        time_values = time_steps.tolist()
        if self.sampler == "sde":
            # Compute fp32 deltas on device once, preserving the established update values.
            time_deltas = (time_steps[1:] - time_steps[:-1]).tolist()
        for step in tqdm(range(num_steps), desc=desc, disable=not sys.stderr.isatty()):
            t_value = time_values[step]
            if self.sampler == "ode":
                x_r, x_i = self._ode_step(
                    x_r, x_i, t_value, time_values[step + 1], stats,
                    condition=condition, cfg_scale=cfg_scale,
                )
            else:
                x_r, x_i = self._euler_maruyama_step(
                    x_r, x_i, t_value, time_deltas[step], stats,
                    condition=condition, cfg_scale=cfg_scale,
                    add_noise=step < num_steps - 1,
                )
        return x_r, x_i, stats
