"""MCTS vs Best-of-N comparison experiment.

Best-of-N: x_0^best = argmin_{i=1..N} R(Solver(Decoder(DDIM(x_T, ε_i))))
MCTS: max_{trajectory τ} E[R(x_0) | trajectory τ chosen]
"""

import torch
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist
from ..core.diffusion.sampler import sample_ddim
from ..core.diffusion.schedule import DDIMSchedule, DDPMSchedule
from ..core.decoding.potential_decoder import PotentialDecoder
from ..core.solver.shortest_path import ShortestPathSolver
from ..core.reward.reward import RewardFunction
from ..core.routing.constraints import Constraints
from .compute_parity import ComputeTracker, count_unet_calls_ddim, count_unet_calls_mcts
from .statistical_tests import statistical_significance_test


def best_of_n_sample(
    model: torch.nn.Module,
    schedule: DDIMSchedule,
    decoder: PotentialDecoder,
    solver: ShortestPathSolver,
    reward_fn: RewardFunction,
    grid: Grid,
    netlist: Netlist,
    n_samples: int,
    T: int = 1000,
    seed: Optional[int] = None
) -> Tuple[float, Dict]:
    """Best-of-N baseline: run DDIM N times, pick best.
    
    Args:
        model: Diffusion model
        schedule: DDIM schedule
        decoder: Potential decoder
        solver: Routing solver
        reward_fn: Reward function
        grid: Grid structure
        netlist: Netlist
        n_samples: Number of independent samples
        T: Total timesteps
        seed: Random seed
    
    Returns:
        (best_reward, metadata) where metadata contains all rewards and compute
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    tracker = ComputeTracker(method_name="best_of_n")
    start_time = time.time()
    
    rewards = []
    for i in range(n_samples):
        sample_seed = seed + i if seed is not None else None
        
        # Sample with DDIM
        shape = (1, 1, grid.height, grid.width)
        x_0 = sample_ddim(
            model, shape, schedule, num_steps=T, eta=0.0, seed=sample_seed
        )
        
        # Track UNet calls
        tracker.add_unet_call(count_unet_calls_ddim(T))
        
        # Decode and solve
        potentials = decoder.decode(x_0.squeeze(0), grid, netlist)
        routing_state = solver.solve(potentials, grid, netlist)
        
        # Compute reward
        constraints = Constraints()
        reward = reward_fn.compute(routing_state, constraints)
        rewards.append(reward)
    
    tracker.add_time(time.time() - start_time)
    
    # Best reward
    best_reward = max(rewards)
    
    metadata = {
        'all_rewards': rewards,
        'best_reward': best_reward,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'compute': tracker.get_summary()
    }
    
    return best_reward, metadata


def mcts_sample(
    mcts_search,
    grid: Grid,
    netlist: Netlist,
    reward_fn: RewardFunction,
    T: int = 1000,
    num_iterations: int = 100,
    seed: Optional[int] = None
) -> Tuple[float, Dict]:
    """MCTS-guided sampling.
    
    Args:
        mcts_search: MCTSSearch instance
        grid: Grid structure
        netlist: Netlist
        reward_fn: Reward function
        T: Total timesteps
        num_iterations: Number of MCTS iterations
        seed: Random seed
    
    Returns:
        (best_reward, metadata)
    """
    from ..core.routing.constraints import Constraints
    
    tracker = ComputeTracker(method_name="mcts")
    start_time = time.time()
    
    # Initialize root
    root_latent = torch.randn(1, grid.height, grid.width)
    
    # Run MCTS search
    best_node = mcts_search.search(
        root_latent.squeeze(0),
        T=T,
        num_simulations=num_iterations,
        seed=seed
    )
    
    # Estimate UNet calls (simplified - actual count would be tracked in search)
    avg_rollout_depth = T // 2  # Estimate
    tracker.add_unet_call(count_unet_calls_mcts(num_iterations, avg_rollout_depth))
    tracker.add_time(time.time() - start_time)
    
    # Decode best node and compute reward
    if best_node.is_terminal():
        potentials = mcts_search.decoder(best_node.latent, grid, netlist)
        routing_state = mcts_search.solver(potentials, grid, netlist)
        constraints = Constraints()
        reward = reward_fn.compute(routing_state, constraints)
    else:
        # Use Q-value as proxy
        reward = best_node.q_value
    
    metadata = {
        'reward': reward,
        'q_value': best_node.q_value,
        'visit_count': best_node.visit_count,
        'compute': tracker.get_summary()
    }
    
    return reward, metadata


def compare_mcts_vs_bestofn(
    model: torch.nn.Module,
    schedule: DDIMSchedule,
    decoder: PotentialDecoder,
    solver: ShortestPathSolver,
    reward_fn: RewardFunction,
    mcts_search,
    grid: Grid,
    netlist: Netlist,
    n_samples: int = 30,
    mcts_iterations: int = 100,
    T: int = 1000,
    seed: Optional[int] = None,
    target_unet_calls: Optional[int] = None
) -> Dict:
    """Compare MCTS vs Best-of-N with compute parity.
    
    Args:
        model: Diffusion model
        schedule: DDIM schedule
        decoder: Potential decoder
        solver: Routing solver
        reward_fn: Reward function
        mcts_search: MCTSSearch instance
        grid: Grid structure
        netlist: Netlist
        n_samples: Number of samples for Best-of-N
        mcts_iterations: Number of MCTS iterations
        T: Total timesteps
        seed: Random seed
        target_unet_calls: Target UNet calls (if None, uses Best-of-N count)
    
    Returns:
        Comparison results with statistical significance
    """
    from .compute_parity import ComputeParityManager
    
    parity_manager = ComputeParityManager(target_unet_calls=target_unet_calls)
    
    # Run Best-of-N
    bestofn_rewards = []
    bestofn_metadata_list = []
    for i in range(n_samples):
        _, metadata = best_of_n_sample(
            model, schedule, decoder, solver, reward_fn,
            grid, netlist, n_samples=1, T=T, seed=seed + i if seed else None
        )
        bestofn_rewards.append(metadata['best_reward'])
        bestofn_metadata_list.append(metadata)
    
    # Determine target compute from Best-of-N
    if target_unet_calls is None:
        target_unet_calls = bestofn_metadata_list[0]['compute']['unet_calls']
    
    # Run MCTS with same compute budget
    mcts_rewards = []
    mcts_metadata_list = []
    for i in range(n_samples):
        # Adjust iterations to match compute
        # Estimate: each MCTS iteration uses ~(1 + rollout_depth) UNet calls
        avg_rollout_depth = T // 2
        calls_per_iteration = 1 + avg_rollout_depth
        adjusted_iterations = target_unet_calls // calls_per_iteration
        
        _, metadata = mcts_sample(
            mcts_search, grid, netlist, reward_fn,
            T=T, num_iterations=adjusted_iterations,
            seed=seed + i if seed else None
        )
        mcts_rewards.append(metadata['reward'])
        mcts_metadata_list.append(metadata)
    
    # Statistical test
    test_results = statistical_significance_test(
        mcts_rewards, bestofn_rewards, min_samples=n_samples
    )
    
    return {
        'mcts_rewards': mcts_rewards,
        'bestofn_rewards': bestofn_rewards,
        'mcts_mean': test_results['mcts_mean'],
        'bestofn_mean': test_results['baseline_mean'],
        'mcts_std': test_results['mcts_std'],
        'bestofn_std': test_results['baseline_std'],
        'statistical_test': test_results,
        'compute_parity': {
            'target_unet_calls': target_unet_calls,
            'mcts_avg_calls': np.mean([m['compute']['unet_calls'] for m in mcts_metadata_list]),
            'bestofn_avg_calls': np.mean([m['compute']['unet_calls'] for m in bestofn_metadata_list])
        }
    }

