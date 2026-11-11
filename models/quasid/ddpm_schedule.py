import torch
from typing import Optional


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather scalar schedule values at timesteps `t: (B,)` into `(B, 1, 1)`."""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def gather_with_terminal(consts: torch.Tensor, t: torch.Tensor, terminal_value: float):
    """Gather schedule values and use `terminal_value` for virtual timestep -1."""
    safe_t = torch.clamp(t, min=0)
    values = gather(consts, safe_t)
    terminal = torch.full_like(values, terminal_value)
    return torch.where(t.reshape(-1, 1, 1) < 0, terminal, values)


class DDPM:
    """Fixed DDPM schedule utilities; QuaSiD wrappers own model loss and spectral covariance."""

    def __init__(self, total_steps: int, device: torch.device):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, total_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.total_steps = total_steps
        self.sigma2 = self.beta #sigma^2 = beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """Return forward DDPM mean and variance for `q(x_t | x_0)`."""
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean.to(self.device), var.to(self.device)

    def q_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Return DDPM forward noising scale `sqrt(1 - alpha_bar_t)` shaped as `(B, 1, 1)`."""
        return torch.sqrt(1 - gather(self.alpha_bar, t)).to(self.device)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Return exact posterior-KL epsilon weight for code timesteps `t >= 1` (paper `t >= 2`)."""
        if (t < 1).any():
            raise ValueError("DDPM posterior-KL weight is defined only for code timesteps t >= 1.")
        alpha = gather(self.alpha, t)
        alpha_bar_prev = gather(self.alpha_bar, t - 1)
        return (1.0 - alpha) / (2.0 * alpha * (1.0 - alpha_bar_prev))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> [torch.Tensor, torch.Tensor]:
        """Sample `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps`."""
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)
        mean, var = self.q_xt_x0(x0, t)
        return (mean + (var**0.5)*eps).to(self.device)

    def p_sample(self, xt: torch.Tensor, n_xt: torch.Tensor, t: torch.Tensor, noise_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classic one-step sampler `x_t -> x_{t-1}` kept as the non-jump DDPM reference."""
        # n_xt = pred_noise_xt = eps_theta
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * n_xt)

        if (t == 0).all():
            return mean

        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        if noise_scale is not None:
            eps = eps * noise_scale
        return mean + (var ** .5) * eps

    def ddim_sample(
        self,
        xt: torch.Tensor,
        eps_theta: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        noise_scale: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generalized DDIM jump step from timestep `t` to `t_next`."""
        alpha_bar_t = gather(self.alpha_bar, t)
        alpha_bar_next = gather_with_terminal(self.alpha_bar, t_next, terminal_value=1.0)

        pred_x0 = (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)

        eta = float(eta)
        if eta > 0:
            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t), min=0.0)
            ) * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_next, min=0.0))
        else:
            sigma = torch.zeros_like(alpha_bar_t)

        eps_coef = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma.pow(2), min=0.0))
        x_next = torch.sqrt(alpha_bar_next) * pred_x0 + eps_coef * eps_theta

        if eta > 0 and not (t_next < 0).all():
            noise = torch.randn_like(xt)
            if noise_scale is not None:
                noise = noise * noise_scale
            x_next = x_next + sigma * noise

        return x_next
