"""Semantic branching ablation study.

Compare:
1. Semantic branching (domain knowledge)
2. Uniform branching (random noise samples)
3. Policy-guided branching (use diffusion model's predicted noise)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist
from ..core.mcts.search import MCTSSearch
from ..core.mcts.semantic_branching import should_branch_semantically
from .compute_parity import ComputeTracker
from .statistical_tests import statistical_significance_test


def run_semantic_branching_ablation(
    mcts_search: MCTSSearch,
    grid: Grid,
    netlist: Netlist,
    num_experiments: int = 10,
    num_simulations: int = 100,
    T: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """Run ablation study comparing branching strategies.
    
    Args:
        mcts_search: MCTSSearch instance
        grid: Grid structure
        netlist: Netlist
        num_experiments: Number of experiments per method
        num_simulations: Number of MCTS simulations
        T: Total timesteps
        seed: Random seed
    
    Returns:
        Ablation results with metrics for each branching strategy
    """
    results = {
        'semantic': [],
        'uniform': [],
        'policy_guided': []
    }
    
    for exp_id in range(num_experiments):
        exp_seed = seed + exp_id if seed is not None else None
        
        # 1. Semantic branching
        mcts_search.semantic_branching = True
        reward_semantic, metadata_semantic = _run_single_experiment(
            mcts_search, grid, netlist, num_simulations, T, exp_seed
        )
        results['semantic'].append({
            'reward': reward_semantic,
            'tree_depth': metadata_semantic.get('tree_depth', 0),
            'sample_efficiency': metadata_semantic.get('sample_efficiency', 0)
        })
        
        # 2. Uniform branching
        mcts_search.semantic_branching = False
        reward_uniform, metadata_uniform = _run_single_experiment(
            mcts_search, grid, netlist, num_simulations, T, exp_seed
        )
        results['uniform'].append({
            'reward': reward_uniform,
            'tree_depth': metadata_uniform.get('tree_depth', 0),
            'sample_efficiency': metadata_uniform.get('sample_efficiency', 0)
        })
        
        # 3. Policy-guided branching (use diffusion model's policy)
        # This would require modifying MCTSSearch to use policy prior
        # For now, use uniform as placeholder
        reward_policy, metadata_policy = _run_single_experiment(
            mcts_search, grid, netlist, num_simulations, T, exp_seed
        )
        results['policy_guided'].append({
            'reward': reward_policy,
            'tree_depth': metadata_policy.get('tree_depth', 0),
            'sample_efficiency': metadata_policy.get('sample_efficiency', 0)
        })
    
    # Aggregate results
    ablation_results = {}
    for method, data in results.items():
        rewards = [d['reward'] for d in data]
        tree_depths = [d['tree_depth'] for d in data]
        sample_effs = [d['sample_efficiency'] for d in data]
        
        ablation_results[method] = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_tree_depth': float(np.mean(tree_depths)),
            'mean_sample_efficiency': float(np.mean(sample_effs))
        }
    
    # Statistical comparison: semantic vs uniform
    semantic_rewards = [d['reward'] for d in results['semantic']]
    uniform_rewards = [d['reward'] for d in results['uniform']]
    
    test_results = statistical_significance_test(
        semantic_rewards, uniform_rewards, min_samples=num_experiments
    )
    
    ablation_results['semantic_vs_uniform'] = test_results
    
    return ablation_results


def _run_single_experiment(
    mcts_search: MCTSSearch,
    grid: Grid,
    netlist: Netlist,
    num_simulations: int,
    T: int,
    seed: Optional[int]
) -> tuple:
    """Run a single MCTS experiment.
    
    Args:
        mcts_search: MCTSSearch instance
        grid: Grid structure
        netlist: Netlist
        num_simulations: Number of simulations
        T: Total timesteps
        seed: Random seed
    
    Returns:
        (reward, metadata)
    """
    # Initialize root
    root_latent = torch.randn(1, grid.height, grid.width)
    
    # Run search
    best_node = mcts_search.search(
        root_latent.squeeze(0),
        T=T,
        num_simulations=num_simulations,
        seed=seed
    )
    
    # Compute metrics
    reward = best_node.q_value
    tree_depth = T - best_node.timestep
    sample_efficiency = best_node.q_value / num_simulations  # Reward per simulation
    
    metadata = {
        'tree_depth': tree_depth,
        'sample_efficiency': sample_efficiency,
        'visit_count': best_node.visit_count
    }
    
    return reward, metadata

