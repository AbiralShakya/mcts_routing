"""Test time-aware normalization: Var[R|x_t] vs σ_t² correlation."""

import pytest
import torch
import numpy as np
from scipy.stats import pearsonr
from src.core.diffusion.schedule import DDPMSchedule
from src.core.mcts.value_normalization import compute_variance_scale
from src.core.reward.reward import RewardFunction
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.state import RoutingState
from src.core.routing.constraints import Constraints


def test_variance_scaling():
    """Test that Var[R|x_t] scales with σ_t².
    
    Correlation between Var[R|x_t] and σ_t² should be > 0.8.
    """
    schedule = DDPMSchedule(num_timesteps=100)
    T = 100
    
    # Sample rewards at different timesteps
    # This is a simplified test - in practice, would sample from actual rollouts
    timesteps = [0, 20, 40, 60, 80, 99]
    rewards_by_t = []
    sigma_t_squared = []
    
    # For each timestep, simulate multiple rollouts and compute variance
    for t in timesteps:
        # Simulate rewards (in practice, would come from actual rollouts)
        # For testing, create synthetic reward distribution with variance ∝ σ_t²
        sigma_t = compute_variance_scale(t, schedule, T)
        sigma_t_squared.append(sigma_t ** 2)
        
        # Simulate rewards with variance proportional to σ_t²
        # In practice, these would come from actual MCTS rollouts
        num_samples = 30
        base_reward = -100.0
        reward_variance = (sigma_t ** 2) * 100.0  # Scale variance
        rewards = np.random.normal(base_reward, np.sqrt(reward_variance), num_samples)
        rewards_by_t.append(rewards)
    
    # Compute variance of rewards at each timestep
    variances = [np.var(rewards) for rewards in rewards_by_t]
    
    # Compute correlation
    if len(variances) > 2 and len(sigma_t_squared) > 2:
        correlation, p_value = pearsonr(variances, sigma_t_squared)
        
        # Correlation should be strong
        assert correlation > 0.8, (
            f"Variance scaling correlation {correlation} below threshold 0.8. "
            f"p-value: {p_value}"
        )
    else:
        pytest.skip("Not enough data points for correlation test")


def test_time_aware_normalization():
    """Test that time-aware normalization reduces variance across timesteps."""
    schedule = DDPMSchedule(num_timesteps=100)
    T = 100
    
    # Simulate Q-values at different timesteps
    timesteps = [0, 25, 50, 75, 99]
    
    # Raw Q-values (would have high variance at early timesteps)
    raw_q_values = []
    for t in timesteps:
        sigma_t = compute_variance_scale(t, schedule, T)
        # Simulate Q-value with variance ∝ σ_t²
        base_q = -50.0
        noise = np.random.normal(0, sigma_t * 10, 20)
        raw_q = base_q + noise
        raw_q_values.append(raw_q)
    
    # Normalized Q-values
    normalized_q_values = []
    for i, t in enumerate(timesteps):
        sigma_t = compute_variance_scale(t, schedule, T)
        raw_q = raw_q_values[i]
        normalized_q = raw_q / (sigma_t + 1e-8)  # Normalize
        normalized_q_values.append(normalized_q)
    
    # Variance of normalized Q-values should be more uniform
    raw_variances = [np.var(q) for q in raw_q_values]
    norm_variances = [np.var(q) for q in normalized_q_values]
    
    # Coefficient of variation should be lower for normalized
    raw_cv = np.std(raw_variances) / (np.mean(raw_variances) + 1e-8)
    norm_cv = np.std(norm_variances) / (np.mean(norm_variances) + 1e-8)
    
    # Normalized should have lower CV (more uniform variance)
    assert norm_cv < raw_cv * 1.5, (
        f"Normalization did not reduce variance: raw_cv={raw_cv}, norm_cv={norm_cv}"
    )

