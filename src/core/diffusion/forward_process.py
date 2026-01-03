"""Forward diffusion process for training."""

import torch
from typing import Tuple

from .schedule import NoiseSchedule, DDPMSchedule


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor = None
) -> torch.Tensor:
    """Forward diffusion process: q(x_t | x_0).
    
    Args:
        x_0: Clean data [B, C, H, W]
        t: Timesteps [B]
        schedule: Noise schedule
        noise: Optional noise tensor (for determinism)
    
    Returns:
        x_t: Noisy data [B, C, H, W]
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    if isinstance(schedule, DDPMSchedule):
        # DDPM: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        alpha_cumprod_t = schedule.get_alpha_cumprod_t(t)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod_t)
        
        # Expand for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    else:
        # Generic schedule
        alpha_t = schedule.get_alpha_t(t)
        sigma_t = schedule.get_sigma_t(t)
        
        while len(alpha_t.shape) < len(x_0.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        while len(sigma_t.shape) < len(x_0.shape):
            sigma_t = sigma_t.unsqueeze(-1)
        
        x_t = alpha_t * x_0 + sigma_t * noise
    
    return x_t

