"""Noise schedules for diffusion (DDPM, DDIM, SDE)."""

import torch
import numpy as np
from typing import Literal


class NoiseSchedule:
    """Base class for noise schedules."""
    
    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t (signal scale)."""
        raise NotImplementedError
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta_t (noise scale)."""
        raise NotImplementedError
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t (noise standard deviation)."""
        raise NotImplementedError


class DDPMSchedule(NoiseSchedule):
    """DDPM noise schedule."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: Literal["linear", "cosine"] = "linear"
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Precompute betas
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Precompute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute sqrt values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, num_timesteps: int) -> torch.Tensor:
        """Cosine beta schedule."""
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t."""
        return self.alphas[t]
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta_t."""
        return self.betas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t (sqrt(1 - alpha_cumprod_t))."""
        return self.sqrt_one_minus_alphas_cumprod[t]
    
    def get_alpha_cumprod_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative product of alphas."""
        return self.alphas_cumprod[t]
    
    def get_noise_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise variance at timestep t (for value normalization)."""
        return 1.0 - self.alphas_cumprod[t]


class DDIMSchedule(NoiseSchedule):
    """DDIM schedule (uses same betas as DDPM but different sampling)."""
    
    def __init__(self, ddpm_schedule: DDPMSchedule, eta: float = 0.0):
        """Initialize DDIM schedule from DDPM schedule.
        
        Args:
            ddpm_schedule: Base DDPM schedule
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
        """
        super().__init__(ddpm_schedule.num_timesteps)
        self.ddpm_schedule = ddpm_schedule
        self.eta = eta
        
        # Precompute DDIM variances
        self.alphas_cumprod = ddpm_schedule.alphas_cumprod
        self.alphas_cumprod_prev = ddpm_schedule.alphas_cumprod_prev
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t."""
        return self.ddpm_schedule.get_alpha_t(t)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta_t."""
        return self.ddpm_schedule.get_beta_t(t)
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t for DDIM."""
        # DDIM uses different variance
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]
        sigma_t = self.eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )
        return sigma_t
    
    def get_noise_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise variance at timestep t."""
        return 1.0 - self.alphas_cumprod[t]


class SDESchedule(NoiseSchedule):
    """SDE (Stochastic Differential Equation) schedule."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        sde_type: Literal["vpsde", "vpsde_linear"] = "vpsde"
    ):
        super().__init__(num_timesteps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sde_type = sde_type
        
        # Precompute sigmas
        if sde_type == "vpsde":
            # Variance preserving SDE
            t = torch.linspace(0, 1, num_timesteps)
            self.sigmas = sigma_min * (sigma_max / sigma_min) ** t
        elif sde_type == "vpsde_linear":
            self.sigmas = torch.linspace(sigma_min, sigma_max, num_timesteps)
        else:
            raise ValueError(f"Unknown SDE type: {sde_type}")
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_t for SDE."""
        # For SDE, alpha_t = 1 / sqrt(1 + sigma_t^2)
        sigma_t = self.get_sigma_t(t)
        return 1.0 / torch.sqrt(1.0 + sigma_t ** 2)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta_t for SDE."""
        # Not directly used in SDE
        return torch.zeros_like(t, dtype=torch.float32)
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma_t."""
        return self.sigmas[t]
    
    def get_noise_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise variance at timestep t."""
        sigma_t = self.get_sigma_t(t)
        return sigma_t ** 2

