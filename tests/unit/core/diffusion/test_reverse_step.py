"""Test reverse diffusion steps."""

import pytest
import torch
from src.core.diffusion.schedule import DDPMSchedule, DDIMSchedule
from src.core.diffusion.sampler import reverse_step_ddpm, reverse_step_ddim


def test_reverse_step_ddpm():
    """Test DDPM reverse step."""
    schedule = DDPMSchedule(num_timesteps=100)
    
    x_t = torch.randn(2, 1, 8, 8)
    predicted_noise = torch.randn(2, 1, 8, 8)
    t = torch.tensor([50, 75])
    
    x_t_prev = reverse_step_ddpm(x_t, predicted_noise, t, schedule)
    
    assert x_t_prev.shape == x_t.shape
    assert not torch.isnan(x_t_prev).any()
    assert not torch.isinf(x_t_prev).any()


def test_reverse_step_ddim():
    """Test DDIM reverse step."""
    ddpm_schedule = DDPMSchedule(num_timesteps=100)
    ddim_schedule = DDIMSchedule(ddpm_schedule, eta=0.0)
    
    x_t = torch.randn(2, 1, 8, 8)
    predicted_noise = torch.randn(2, 1, 8, 8)
    t = torch.tensor([50, 75])
    t_prev = torch.tensor([49, 74])
    
    x_t_prev = reverse_step_ddim(x_t, predicted_noise, t, t_prev, ddim_schedule, eta=0.0)
    
    assert x_t_prev.shape == x_t.shape
    assert not torch.isnan(x_t_prev).any()
    assert not torch.isinf(x_t_prev).any()

