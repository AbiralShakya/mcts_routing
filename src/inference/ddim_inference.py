"""DDIM inference baseline."""

import torch
from typing import Dict, Any, Optional
from ..core.diffusion.sampler import sample_ddim
from ..core.diffusion.schedule import DDIMSchedule
from ..core.decoding.potential_decoder import PotentialDecoder
from ..core.solver.shortest_path import ShortestPathSolver
from ..core.reward.reward import RewardFunction
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist
from ..core.routing.constraints import Constraints
from ..comparison.compute_parity import ComputeTracker, count_unet_calls_ddim


def ddim_inference(
    model: torch.nn.Module,
    schedule: DDIMSchedule,
    decoder: PotentialDecoder,
    solver: ShortestPathSolver,
    reward_fn: RewardFunction,
    grid: Grid,
    netlist: Netlist,
    T: int = 1000,
    eta: float = 0.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run DDIM inference.
    
    Args:
        model: Diffusion model
        schedule: DDIM schedule
        decoder: Potential decoder
        solver: Routing solver
        reward_fn: Reward function
        grid: Grid structure
        netlist: Netlist
        T: Total timesteps
        eta: DDIM eta parameter (0.0 = deterministic)
        seed: Random seed
    
    Returns:
        Dictionary with results:
        {
            'reward': float,
            'routing_state': RoutingState,
            'potentials': RoutingPotentials,
            'compute': Dict
        }
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    tracker = ComputeTracker(method_name="ddim")
    import time
    start_time = time.time()
    
    # Sample with DDIM
    shape = (1, 1, grid.height, grid.width)
    x_0 = sample_ddim(
        model, shape, schedule, num_steps=T, eta=eta, seed=seed
    )
    
    # Track compute
    tracker.add_unet_call(count_unet_calls_ddim(T))
    tracker.add_time(time.time() - start_time)
    
    # Decode to potentials
    potentials = decoder.decode(x_0.squeeze(0), grid, netlist)
    
    # Solve to get routing
    routing_state = solver.solve(potentials, grid, netlist)
    
    # Compute reward
    constraints = Constraints()
    reward = reward_fn.compute(routing_state, constraints)
    
    return {
        'reward': float(reward),
        'routing_state': routing_state,
        'potentials': potentials,
        'compute': tracker.get_summary()
    }
