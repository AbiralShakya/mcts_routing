"""Diffusion sampling procedures (DDPM, DDIM)."""

import torch
from typing import Optional

from .schedule import NoiseSchedule, DDPMSchedule, DDIMSchedule


def reverse_step_ddpm(
    x_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule
) -> torch.Tensor:
    """DDPM reverse step: x_{t-1} = f(x_t, predicted_noise, t).
    
    Args:
        x_t: Noisy data at timestep t [B, C, H, W]
        predicted_noise: Predicted noise [B, C, H, W]
        t: Timestep [B]
        schedule: DDPM schedule
    
    Returns:
        x_{t-1}: Denoised data [B, C, H, W]
    """
    alpha_cumprod_t = schedule.get_alpha_cumprod_t(t)
    alpha_cumprod_t_prev = schedule.alphas_cumprod_prev[t]
    beta_t = schedule.get_beta_t(t)
    
    # Expand for broadcasting
    while len(alpha_cumprod_t.shape) < len(x_t.shape):
        alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
    while len(alpha_cumprod_t_prev.shape) < len(x_t.shape):
        alpha_cumprod_t_prev = alpha_cumprod_t_prev.unsqueeze(-1)
    while len(beta_t.shape) < len(x_t.shape):
        beta_t = beta_t.unsqueeze(-1)
    
    # Predict x_0
    pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    # Compute x_{t-1}
    pred_dir = torch.sqrt(1.0 - alpha_cumprod_t_prev) * predicted_noise
    random_noise = torch.randn_like(x_t)
    x_t_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x_0 + pred_dir + torch.sqrt(beta_t) * random_noise
    
    return x_t_prev


def reverse_step_ddim(
    x_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    schedule: DDIMSchedule,
    eta: float = 0.0
) -> torch.Tensor:
    """DDIM reverse step (deterministic when eta=0).
    
    Args:
        x_t: Noisy data at timestep t [B, C, H, W]
        predicted_noise: Predicted noise [B, C, H, W]
        t: Current timestep [B]
        t_prev: Previous timestep [B]
        schedule: DDIM schedule
        eta: DDIM parameter (0 = deterministic)
    
    Returns:
        x_{t_prev}: Denoised data [B, C, H, W]
    """
    alpha_cumprod_t = schedule.alphas_cumprod[t]
    alpha_cumprod_t_prev = schedule.alphas_cumprod_prev[t_prev]
    
    # Expand for broadcasting
    while len(alpha_cumprod_t.shape) < len(x_t.shape):
        alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
    while len(alpha_cumprod_t_prev.shape) < len(x_t.shape):
        alpha_cumprod_t_prev = alpha_cumprod_t_prev.unsqueeze(-1)
    
    # Predict x_0
    pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    # Direction pointing to x_t
    dir_xt = torch.sqrt(1.0 - alpha_cumprod_t_prev - eta ** 2 * (1.0 - alpha_cumprod_t_prev)) * predicted_noise
    
    # Random noise (zero if eta=0, making it deterministic)
    random_noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
    noise_scale = eta * torch.sqrt(1.0 - alpha_cumprod_t_prev)
    
    x_t_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x_0 + dir_xt + noise_scale * random_noise
    
    return x_t_prev


def sample_ddpm(
    model: torch.nn.Module,
    shape: tuple,
    schedule: DDPMSchedule,
    conditioning: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Sample from DDPM model.
    
    Args:
        model: Diffusion model
        shape: Output shape [B, C, H, W]
        schedule: DDPM schedule
        conditioning: Optional conditioning tensor
        num_steps: Number of sampling steps (default: schedule.num_timesteps)
        seed: Random seed for reproducibility
    
    Returns:
        x_0: Sampled data [B, C, H, W]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_steps = num_steps or schedule.num_timesteps
    device = next(model.parameters()).device
    
    # Start from noise
    x_t = torch.randn(shape, device=device)
    
    # Sample
    for i in range(num_steps - 1, -1, -1):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model(x_t, t, conditioning)
        
        # Reverse step
        x_t = reverse_step_ddpm(x_t, predicted_noise, t, schedule)
    
    return x_t


def sample_ddim(
    model: torch.nn.Module,
    shape: tuple,
    schedule: DDIMSchedule,
    conditioning: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
    eta: float = 0.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Sample from DDIM model (deterministic when eta=0).
    
    Args:
        model: Diffusion model
        shape: Output shape [B, C, H, W]
        schedule: DDIM schedule
        conditioning: Optional conditioning tensor
        num_steps: Number of sampling steps
        eta: DDIM parameter (0 = deterministic)
        seed: Random seed (only used if eta > 0)
    
    Returns:
        x_0: Sampled data [B, C, H, W]
    """
    if seed is not None and eta > 0:
        torch.manual_seed(seed)
    
    num_steps = num_steps or schedule.num_timesteps
    device = next(model.parameters()).device
    
    # Start from noise
    x_t = torch.randn(shape, device=device)
    
    # Create timestep sequence
    step_size = schedule.num_timesteps // num_steps
    timesteps = list(range(schedule.num_timesteps - 1, -1, -step_size))
    
    # Sample
    for i in range(len(timesteps) - 1):
        t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
        t_prev = torch.full((shape[0],), timesteps[i + 1], device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model(x_t, t, conditioning)
        
        # Reverse step
        x_t = reverse_step_ddim(x_t, predicted_noise, t, t_prev, schedule, eta)
    
    return x_t

