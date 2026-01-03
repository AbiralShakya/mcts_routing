"""Time-aware value normalization.

Mathematical justification:
Var[R(x_0) | x_t] ∝ σ_t^2 (noise variance at timestep t)
Z_t = sqrt(Var[R | x_t]) normalizes variance across timesteps
"""

import torch
from typing import Optional

from ..diffusion.schedule import NoiseSchedule, DDPMSchedule, DDIMSchedule


def normalize_q_value(
    q_raw: float,
    timestep: int,
    schedule: NoiseSchedule,
    T: int
) -> float:
    """Normalize Q-value by timestep-dependent variance scale.
    
    Args:
        q_raw: Raw Q-value
        timestep: Current timestep
        schedule: Noise schedule
        T: Total number of timesteps
    
    Returns:
        Normalized Q-value
    """
    z_t = compute_variance_scale(timestep, schedule, T)
    if z_t > 0:
        return q_raw / z_t
    return q_raw


def compute_variance_scale(
    t: int,
    schedule: NoiseSchedule,
    T: int
) -> float:
    """Compute variance scale based on diffusion noise schedule.
    
    σ_t^2 = 1 - α_t (for DDPM) or from SDE schedule
    
    Args:
        t: Timestep
        schedule: Noise schedule
        T: Total timesteps
    
    Returns:
        Variance scale (standard deviation)
    """
    t_tensor = torch.tensor([t])
    variance = schedule.get_noise_variance(t_tensor)
    return float(variance.sqrt().item())


def compute_variance_scale_batch(
    timesteps: torch.Tensor,
    schedule: NoiseSchedule,
    T: int
) -> torch.Tensor:
    """Compute variance scale for batch of timesteps.
    
    Args:
        timesteps: Timesteps [B]
        schedule: Noise schedule
        T: Total timesteps
    
    Returns:
        Variance scales [B]
    """
    variances = schedule.get_noise_variance(timesteps)
    return variances.sqrt()

