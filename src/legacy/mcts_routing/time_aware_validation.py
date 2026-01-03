"""Empirical validation of time-aware normalization.

Measure Var[R|x_t] vs σ_t², plot correlation, compare with/without normalization.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import pearsonr
from ..diffusion.schedule import NoiseSchedule, DDPMSchedule
from ..diffusion.sampler import sample_ddpm, sample_ddim
from ..decoding.potential_decoder import PotentialDecoder
from ..solver.shortest_path import ShortestPathSolver
from ..reward.reward import RewardFunction
from ..routing.grid import Grid
from ..routing.netlist import Netlist
from ..routing.constraints import Constraints
from .value_normalization import compute_variance_scale
from .search import MCTSSearch


def measure_reward_variance_by_timestep(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    decoder: PotentialDecoder,
    solver: ShortestPathSolver,
    reward_fn: RewardFunction,
    grid: Grid,
    netlist: Netlist,
    T: int = 1000,
    timesteps: Optional[List[int]] = None,
    num_samples_per_t: int = 30,
    seed: Optional[int] = None
) -> Dict:
    """Measure reward variance at different timesteps.
    
    Args:
        model: Diffusion model
        schedule: Noise schedule
        decoder: Potential decoder
        solver: Routing solver
        reward_fn: Reward function
        grid: Grid structure
        netlist: Netlist
        T: Total timesteps
        timesteps: List of timesteps to measure (if None, uses [0, T/4, T/2, 3T/4, T-1])
        num_samples_per_t: Number of samples per timestep
        seed: Random seed
    
    Returns:
        Dict with variance measurements and correlation with σ_t²
    """
    if timesteps is None:
        timesteps = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    constraints = Constraints()
    
    # Measure variance at each timestep
    reward_variances = []
    sigma_t_squared_list = []
    
    for t in timesteps:
        # Get noise variance at this timestep
        sigma_t = compute_variance_scale(t, schedule, T)
        sigma_t_squared_list.append(sigma_t ** 2)
        
        # Sample multiple rollouts from this timestep
        rewards = []
        for i in range(num_samples_per_t):
            sample_seed = seed + i if seed is not None else None
            
            # Sample x_t
            if t == T - 1:
                # Pure noise
                x_t = torch.randn(1, grid.height, grid.width)
            else:
                # Sample from forward process (simplified)
                x_0 = torch.randn(1, grid.height, grid.width)  # Placeholder
                # In practice, would use forward process: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
                alpha_t = schedule.get_alpha_t(torch.tensor([t]))[0].item()
                noise = torch.randn_like(x_0)
                x_t = np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * noise
            
            # Complete rollout to t=0
            # Simplified: just decode and solve
            try:
                potentials = decoder.decode(x_t.squeeze(0), grid, netlist)
                routing_state = solver.solve(potentials, grid, netlist)
                reward = reward_fn.compute(routing_state, constraints)
                rewards.append(reward)
            except Exception:
                continue
        
        if len(rewards) > 1:
            variance = np.var(rewards)
            reward_variances.append(variance)
        else:
            reward_variances.append(0.0)
    
    # Compute correlation
    if len(reward_variances) > 2 and len(sigma_t_squared_list) > 2:
        correlation, p_value = pearsonr(reward_variances, sigma_t_squared_list)
    else:
        correlation, p_value = 0.0, 1.0
    
    return {
        'timesteps': timesteps,
        'reward_variances': reward_variances,
        'sigma_t_squared': sigma_t_squared_list,
        'correlation': float(correlation),
        'p_value': float(p_value),
        'correlation_strong': abs(correlation) > 0.8
    }


def compare_with_without_normalization(
    mcts_search_with: MCTSSearch,
    mcts_search_without: MCTSSearch,
    grid: Grid,
    netlist: Netlist,
    num_experiments: int = 30,
    num_simulations: int = 100,
    T: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """Compare MCTS with and without time-aware normalization.
    
    Args:
        mcts_search_with: MCTSSearch with normalization enabled
        mcts_search_without: MCTSSearch with normalization disabled
        grid: Grid structure
        netlist: Netlist
        num_experiments: Number of experiments
        num_simulations: Number of MCTS simulations
        T: Total timesteps
        seed: Random seed
    
    Returns:
        Comparison results
    """
    rewards_with = []
    rewards_without = []
    
    for i in range(num_experiments):
        exp_seed = seed + i if seed is not None else None
        
        # With normalization
        root_latent = torch.randn(1, grid.height, grid.width)
        best_with = mcts_search_with.search(
            root_latent.squeeze(0), T=T, num_simulations=num_simulations, seed=exp_seed
        )
        rewards_with.append(best_with.q_value)
        
        # Without normalization
        root_latent = torch.randn(1, grid.height, grid.width)
        best_without = mcts_search_without.search(
            root_latent.squeeze(0), T=T, num_simulations=num_simulations, seed=exp_seed
        )
        rewards_without.append(best_without.q_value)
    
    # Statistical test
    from ...comparison.statistical_tests import statistical_significance_test
    test_results = statistical_significance_test(
        rewards_with, rewards_without, min_samples=num_experiments
    )
    
    return {
        'with_normalization': {
            'mean': float(np.mean(rewards_with)),
            'std': float(np.std(rewards_with))
        },
        'without_normalization': {
            'mean': float(np.mean(rewards_without)),
            'std': float(np.std(rewards_without))
        },
        'statistical_test': test_results
    }

