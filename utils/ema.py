import copy

import torch


class EMA:
    def __init__(self, model, beta=0.995, update_every=10):
        self.model = model
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        self.initted = False
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device):
        self.ema_model.to(device)
        return self

    @torch.no_grad()
    def copy_params_from_model(self):
        """Synchronize EMA with the current online model before averaging begins."""
        self.ema_model.load_state_dict(self.model.state_dict())
        self.initted = True

    @torch.no_grad()
    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return

        # A deepcopy at construction contains the *initial* model parameters.
        # Use the first scheduled EMA update to establish the correct starting
        # point, then apply exponential averaging on later updates.
        if not self.initted:
            self.copy_params_from_model()
            return

        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.beta).add_(param.data, alpha=1.0 - self.beta)

        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self):
        return {
            "step": self.step,
            "initted": self.initted,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = int(state_dict.get("step", 0))
        ema_state = state_dict.get("ema_model", state_dict)
        self.ema_model.load_state_dict(ema_state)
        # Checkpoints written before this fix already contain a valid EMA
        # tensor state. Treat them as initialized for backward compatibility.
        self.initted = bool(state_dict.get("initted", True))
