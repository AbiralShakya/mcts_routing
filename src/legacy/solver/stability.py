"""Stability mechanisms for solvers (tie-breaking, entropy regularization, soft penalties)."""

import torch
import random
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
from collections import defaultdict

from ..decoding.decoder import RoutingPotentials
from ..routing.state import RoutingState, WireSegment
from ..routing.grid import Grid
from ..routing.netlist import Netlist


class RandomizedTieBreakingSolver:
    """Wrapper solver that runs K times with randomized tie-breaking and averages results."""
    
    def __init__(
        self,
        base_solver,
        k_runs: int = 10,
        aggregation_method: str = "majority_vote"
    ):
        """Initialize randomized tie-breaking solver.
        
        Args:
            base_solver: Base solver to wrap
            k_runs: Number of runs for averaging
            aggregation_method: "majority_vote" or "average_potentials"
        """
        self.base_solver = base_solver
        self.k_runs = k_runs
        self.aggregation_method = aggregation_method
    
    def solve(
        self,
        potentials: RoutingPotentials,
        grid: Grid,
        netlist: Netlist,
        stability_mode: str = "randomized_tiebreak",
        k_runs: Optional[int] = None,
        tiebreak_seed: Optional[int] = None
    ) -> RoutingState:
        """Solve with randomized tie-breaking.
        
        Args:
            potentials: Routing potentials
            grid: Grid structure
            netlist: Netlist
            stability_mode: Stability mode (ignored, always uses tie-breaking)
            k_runs: Number of runs (overrides self.k_runs if provided)
            tiebreak_seed: Random seed
        
        Returns:
            Aggregated routing state
        """
        k = k_runs if k_runs is not None else self.k_runs
        
        routing_states = []
        for i in range(k):
            run_seed = tiebreak_seed + i if tiebreak_seed is not None else None
            state = self.base_solver.solve(
                potentials, grid, netlist,
                stability_mode="randomized_tiebreak",
                k_runs=1,
                tiebreak_seed=run_seed
            )
            routing_states.append(state)
        
        # Aggregate results
        if self.aggregation_method == "majority_vote":
            return self._aggregate_majority_vote(routing_states, grid, netlist)
        elif self.aggregation_method == "average_potentials":
            return self._aggregate_average_potentials(routing_states, grid, netlist)
        else:
            # Default: return first state
            return routing_states[0]
    
    def _aggregate_majority_vote(
        self,
        routing_states: List[RoutingState],
        grid: Grid,
        netlist: Netlist
    ) -> RoutingState:
        """Aggregate routes using majority vote on wire segments.
        
        Args:
            routing_states: List of routing states
            grid: Grid structure
            netlist: Netlist
        
        Returns:
            Aggregated routing state
        """
        # Count wire segment usage
        segment_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], int] = defaultdict(int)
        
        for state in routing_states:
            for wire in state.wires:
                key = (wire.start, wire.end, wire.net_id)
                segment_counts[key] += 1
        
        # Create aggregated state with segments that appear in majority
        aggregated = RoutingState(grid=grid, netlist=netlist)
        threshold = len(routing_states) / 2.0
        
        for (start, end, net_id), count in segment_counts.items():
            if count >= threshold:
                wire = WireSegment(start=start, end=end, net_id=net_id, layer=0)
                aggregated.add_wire(wire)
        
        return aggregated
    
    def _aggregate_average_potentials(
        self,
        routing_states: List[RoutingState],
        grid: Grid,
        netlist: Netlist
    ) -> RoutingState:
        """Aggregate by averaging potential fields and re-solving.
        
        Args:
            routing_states: List of routing states
            grid: Grid structure
            netlist: Netlist
        
        Returns:
            Aggregated routing state
        """
        # Convert routing states to potential fields and average
        from ...data.generation.synthetic import routing_state_to_potentials
        
        potential_fields = []
        for state in routing_states:
            x_0 = routing_state_to_potentials(state, grid)
            potential_fields.append(x_0)
        
        # Average potential fields
        avg_potentials = torch.stack(potential_fields).mean(dim=0)
        
        # Create RoutingPotentials from averaged field
        avg_routing_potentials = RoutingPotentials(
            cost_field=avg_potentials.squeeze(-1),
            edge_weights={}
        )
        
        # Re-solve with averaged potentials
        return self.base_solver.solve(
            avg_routing_potentials, grid, netlist,
            stability_mode="randomized_tiebreak",
            k_runs=1
        )


def randomized_tiebreak_averaging(
    solver_func: Callable,
    potentials: RoutingPotentials,
    grid: Grid,
    netlist: Netlist,
    k_runs: int = 10,
    seed: Optional[int] = None
) -> RoutingState:
    """Run solver K times with randomized tie-breaking and average results.
    
    Args:
        solver_func: Solver function to call
        potentials: Routing potentials
        grid: Grid
        netlist: Netlist
        k_runs: Number of runs
        seed: Random seed
    
    Returns:
        Averaged routing state
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    routing_states = []
    for i in range(k_runs):
        # Use different seed for each run
        run_seed = seed + i if seed is not None else None
        if run_seed is not None:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
        
        state = solver_func(potentials, grid, netlist, tiebreak_seed=run_seed)
        routing_states.append(state)
    
    # Aggregate using majority vote
    wrapper = RandomizedTieBreakingSolver(solver_func, k_runs=1)
    return wrapper._aggregate_majority_vote(routing_states, grid, netlist)


def add_entropy_regularization(
    potentials: RoutingPotentials,
    noise_scale: float = 1e-4
) -> RoutingPotentials:
    """Add small random noise to potentials to prevent exact ties.
    
    Args:
        potentials: Routing potentials
        noise_scale: Scale of noise to add
    
    Returns:
        Regularized potentials
    """
    # Add small noise
    noise = torch.randn_like(potentials.cost_field) * noise_scale
    cost_field = potentials.cost_field + noise
    
    # Recompute edge weights
    edge_weights = {}
    h, w = cost_field.shape
    for y in range(h):
        for x in range(w):
            if x < w - 1:
                edge = ((x, y), (x + 1, y))
                weight = (cost_field[y, x] + cost_field[y, x + 1]) / 2.0
                edge_weights[edge] = weight.item()
            if y < h - 1:
                edge = ((x, y), (x, y + 1))
                weight = (cost_field[y, x] + cost_field[y + 1, x]) / 2.0
                edge_weights[edge] = weight.item()
    
    return RoutingPotentials(
        cost_field=cost_field,
        edge_weights=edge_weights,
        flow_preferences=potentials.flow_preferences
    )


def add_soft_congestion_penalties(
    potentials: RoutingPotentials,
    usage_map: torch.Tensor,
    capacity_map: torch.Tensor,
    penalty_weight: float = 1.0
) -> RoutingPotentials:
    """Add soft congestion penalties instead of hard capacity cuts.
    
    cost(i,j) = base_cost(i,j) + λ_cong · max(0, usage(i,j) - capacity(i,j))²
    
    Args:
        potentials: Routing potentials
        usage_map: Current usage map [H, W]
        capacity_map: Capacity map [H, W]
        penalty_weight: Weight for congestion penalty
    
    Returns:
        Potentials with soft congestion penalties
    """
    cost_field = potentials.cost_field.clone()
    
    # Compute congestion penalty
    congestion = torch.clamp(usage_map - capacity_map, min=0.0)
    congestion_penalty = penalty_weight * (congestion ** 2)
    
    # Add to cost field
    cost_field = cost_field + congestion_penalty
    
    # Normalize
    if cost_field.max() > 0:
        cost_field = cost_field / cost_field.max()
    
    return RoutingPotentials(
        cost_field=cost_field,
        edge_weights=potentials.edge_weights,
        flow_preferences=potentials.flow_preferences
    )
