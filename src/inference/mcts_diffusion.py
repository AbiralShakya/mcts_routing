"""MCTS-guided diffusion inference."""

import torch
from typing import Dict, Any, Optional
from ..core.mcts.search import MCTSSearch
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist
from ..core.reward.reward import RewardFunction
from ..core.routing.constraints import Constraints
from ..comparison.compute_parity import ComputeTracker


def mcts_inference(
    mcts_search: MCTSSearch,
    grid: Grid,
    netlist: Netlist,
    reward_fn: RewardFunction,
    T: int = 1000,
    num_simulations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run MCTS-guided diffusion inference.
    
    Args:
        mcts_search: MCTSSearch instance
        grid: Grid structure
        netlist: Netlist
        reward_fn: Reward function
        T: Total timesteps
        num_simulations: Number of MCTS simulations
        seed: Random seed
    
    Returns:
        Dictionary with results:
        {
            'reward': float,
            'routing_state': RoutingState,
            'potentials': RoutingPotentials,
            'best_node': MCTSNode,
            'compute': Dict
        }
    """
    import time
    tracker = ComputeTracker(method_name="mcts")
    start_time = time.time()
    
    # Initialize root
    root_latent = torch.randn(1, grid.height, grid.width)
    if seed is not None:
        torch.manual_seed(seed)
    
    # Run MCTS search
    best_node = mcts_search.search(
        root_latent.squeeze(0),
        T=T,
        num_simulations=num_simulations,
        seed=seed
    )
    
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
        routing_state = None
        potentials = None
    
    return {
        'reward': float(reward),
        'routing_state': routing_state,
        'potentials': potentials,
        'best_node': best_node,
        'q_value': float(best_node.q_value),
        'visit_count': best_node.visit_count,
        'compute': tracker.get_summary()
    }
