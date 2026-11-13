import sys

import torch
from tqdm.auto import tqdm

from models.quasid.ddpm_schedule import DDPM
from models.quasid.quasid import QuaSiD


class QuaSiD_disc(QuaSiD):
    """Discrete-time DDPM QuaSiD variant.

    Uses the shared QuaSiD frequency-domain interface and owns only the
    discrete schedule: integer `t in [0, max_step)`, DDPM forward noising,
    and DDIM/generalized jump reverse sampling for R/I spectral branches.
    """

    def __init__(
        self,
        seq_length,
        channels,
        max_step=1000,
        sampling_steps=1000,
        beta_min=None,
        beta_max=None,
        time_eps=None,
        sampler="ode",
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
        # Accept SDE-only fields so one YAML schema can
        # switch wrappers through `model.target`; DDPM owns its discrete schedule.
        del beta_min, beta_max, time_eps
        if sampler not in {"sde", "ode"}:
            raise ValueError(f"sampler must be 'sde' or 'ode', got {sampler!r}.")
        super().__init__(
            seq_length=seq_length,
            channels=channels,
            max_step=int(max_step),
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
        self.sampler = sampler
        sampling_name = "DDIM" if sampler == "ode" else "DDPM"
        self.sample_desc = f"{sampling_name} Sampling"
        self.conditional_desc = f"Conditional {sampling_name}"
        self.ddpm = DDPM(self.max_step, device)

    def forward(self, data, **kwargs):
        x, condition = self._prepare_training_batch(data)
        cond_emb = self.project_condition(condition) if condition is not None else None
        if self.channel_indep:
            x = self._reshape_channels_in(x)
        x_r, x_i, stats = self._to_spectrum(x)
        t = torch.randint(1, self.max_step, (x_r.shape[0],), device=self.device).long()

        noise_r, noise_i = self._spectral_noise_like(x_r, stats)
        xt_r = self.ddpm.q_sample(x_r, t, noise_r)
        xt_i = self.ddpm.q_sample(x_i, t, noise_i)

        pred_r, pred_i = self._predict_noise(
            xt_r,
            xt_i,
            t,
            stats,
            noise_scale=self.ddpm.q_noise_scale(t),
            condition=self._training_condition(cond_emb),
        )
        weight = self.ddpm.loss_weight(t) if self.use_loss_weight else None
        return self._noise_loss(pred_r, pred_i, noise_r, noise_i, stats, weight=weight)

    @torch.no_grad()
    def _ddpm_step(self, x_r, x_i, t_value, next_t_value, stats, condition=None, cfg_scale=0.0):
        """Main sampler step: DDPM training schedule with DDIM/generalized jumps."""
        batch = x_r.shape[0]
        t = torch.full((batch,), int(t_value), dtype=torch.long, device=self.device)
        t_next = torch.full((batch,), int(next_t_value), dtype=torch.long, device=self.device)
        eps_r, eps_i = self._predict_noise(
            x_r,
            x_i,
            t,
            stats,
            noise_scale=self.ddpm.q_noise_scale(t),
            condition=condition,
            cfg_scale=cfg_scale,
        )

        next_r = self.ddpm.ddim_sample(
            x_r,
            eps_r,
            t,
            t_next,
            noise_scale=torch.sqrt(stats["var_r"]),
            eta=0.0 if self.sampler == "ode" else 1.0,
        )
        next_i = self.ddpm.ddim_sample(
            x_i,
            eps_i,
            t,
            t_next,
            noise_scale=torch.sqrt(stats["var_i"]),
            eta=0.0 if self.sampler == "ode" else 1.0,
        )
        return next_r, next_i * (stats["var_i"] > 0)

    def _sampling_time_pairs(self, num_steps):
        """Return DDIM jump pairs `(t, t_next)` ending at terminal t=-1."""
        num_steps = max(1, min(int(num_steps), self.max_step))
        if num_steps == self.max_step:
            time_steps = torch.arange(self.max_step - 1, -1, -1, device=self.device, dtype=torch.long)
        else:
            time_steps = torch.linspace(self.max_step - 1, 0, num_steps, device=self.device).long()
            time_steps = torch.unique_consecutive(time_steps)
        next_steps = torch.cat(
            [
                time_steps[1:],
                torch.tensor([-1], device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        return time_steps, next_steps

    @torch.no_grad()
    def _sample_spectrum(self, batch_size, num_steps, condition=None, cfg_scale=0.0, desc="DDPM Sampling"):
        """Sample R/I spectrum by DDIM jumps over the discrete DDPM chain."""
        x_r, x_i, stats = self._initial_spectrum_state(batch_size)
        time_steps, next_steps = self._sampling_time_pairs(num_steps)
        for t_value, next_t_value in tqdm(
            zip(time_steps.tolist(), next_steps.tolist()),
            total=time_steps.numel(),
            desc=desc,
            disable=not sys.stderr.isatty(),
        ):
            x_r, x_i = self._ddpm_step(
                x_r,
                x_i,
                t_value,
                next_t_value,
                stats,
                condition=condition,
                cfg_scale=cfg_scale,
            )
        return x_r, x_i, stats
