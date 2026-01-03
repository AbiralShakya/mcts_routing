"""Test diffusion schedules."""

import pytest
import torch
from src.core.diffusion.schedule import DDPMSchedule, DDIMSchedule, SDESchedule


def test_ddpm_schedule():
    """Test DDPM schedule."""
    schedule = DDPMSchedule(num_timesteps=100, beta_start=0.0001, beta_end=0.02)
    assert schedule.num_timesteps == 100
    assert len(schedule.betas) == 100
    assert len(schedule.alphas) == 100


def test_ddpm_get_alpha_t():
    """Test getting alpha_t."""
    schedule = DDPMSchedule(num_timesteps=100)
    t = torch.tensor([0, 50, 99])
    alpha_t = schedule.get_alpha_t(t)
    assert alpha_t.shape == (3,)
    assert torch.all(alpha_t > 0)
    assert torch.all(alpha_t <= 1)


def test_ddpm_get_noise_variance():
    """Test noise variance computation."""
    schedule = DDPMSchedule(num_timesteps=100)
    t = torch.tensor([0, 50, 99])
    variance = schedule.get_noise_variance(t)
    assert variance.shape == (3,)
    # Variance should increase with timestep
    assert variance[0] < variance[1] < variance[2]


def test_ddim_schedule():
    """Test DDIM schedule."""
    ddpm_schedule = DDPMSchedule(num_timesteps=100)
    ddim_schedule = DDIMSchedule(ddpm_schedule, eta=0.0)
    assert ddim_schedule.num_timesteps == 100


def test_sde_schedule():
    """Test SDE schedule."""
    schedule = SDESchedule(num_timesteps=100, sigma_min=0.01, sigma_max=50.0)
    assert schedule.num_timesteps == 100
    t = torch.tensor([0, 50, 99])
    sigma_t = schedule.get_sigma_t(t)
    assert sigma_t.shape == (3,)
    # Sigma should increase with timestep
    assert sigma_t[0] < sigma_t[1] < sigma_t[2]

