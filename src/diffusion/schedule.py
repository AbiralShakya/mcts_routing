"""Noise schedules for diffusion."""

import torch
from abc import ABC, abstractmethod
from typing import Optional


class NoiseSchedule(ABC):
    """Base class for noise schedules."""

    @abstractmethod
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha (1 - beta) for timestep."""
        pass

    @abstractmethod
    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative product of alphas."""
        pass


class DDPMSchedule(NoiseSchedule):
    """Linear beta schedule from DDPM."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Precompute betas and alphas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> 'DDPMSchedule':
        """Move schedule tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha for timestep."""
        t = t.long().clamp(0, self.num_timesteps - 1)
        # Move to same device as t if needed
        if self.alphas.device != t.device:
            self.to(t.device)
        return self.alphas[t]

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative alpha for timestep."""
        t = t.long().clamp(0, self.num_timesteps - 1)
        # Move to same device as t if needed
        if self.alpha_bars.device != t.device:
            self.to(t.device)
        return self.alpha_bars[t]

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta for timestep."""
        t = t.long().clamp(0, self.num_timesteps - 1)
        # Move to same device as t if needed
        if self.betas.device != t.device:
            self.to(t.device)
        return self.betas[t]
