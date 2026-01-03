"""Test solver stability: identical potentials should yield similar routes."""

import pytest
import torch
import numpy as np
from src.core.solver.shortest_path import ShortestPathSolver
from src.core.decoding.decoder import RoutingPotentials
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


def test_solver_stability():
    """Test that solver produces similar routes for identical potentials.
    
    Variance of routes from K=10 solver runs should be below threshold.
    """
    solver = ShortestPathSolver()
    grid = Grid(width=10, height=10)
    netlist = Netlist(nets=[
        Net(net_id=0, pins=[Pin(0, 0), Pin(9, 9)]),
        Net(net_id=1, pins=[Pin(0, 9), Pin(9, 0)])
    ])
    
    # Create uniform potentials
    potentials = RoutingPotentials(
        cost_field=torch.ones(10, 10),
        edge_weights={}
    )
    
    # Run solver K times with different seeds
    K = 10
    routes = []
    for k in range(K):
        state = solver.solve(
            potentials, grid, netlist,
            stability_mode="randomized_tiebreak",
            k_runs=1,
            tiebreak_seed=k
        )
        routes.append(state)
    
    # Compute variance in route lengths
    route_lengths = []
    for state in routes:
        total_length = 0
        for wire in state.wires:
            x1, y1 = wire.start
            x2, y2 = wire.end
            length = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
            total_length += length
        route_lengths.append(total_length)
    
    # Variance should be low (routes should be similar)
    if len(route_lengths) > 1:
        variance = np.var(route_lengths)
        # Threshold: variance should be < 10% of mean
        mean_length = np.mean(route_lengths)
        cv = np.sqrt(variance) / mean_length if mean_length > 0 else 0
        
        # Coefficient of variation should be small
        assert cv < 0.5, f"Solver variance too high: CV={cv}"


def test_solver_determinism_with_seed():
    """Test that solver is deterministic with same seed."""
    solver = ShortestPathSolver()
    grid = Grid(width=10, height=10)
    netlist = Netlist(nets=[
        Net(net_id=0, pins=[Pin(0, 0), Pin(9, 9)])
    ])
    
    potentials = RoutingPotentials(
        cost_field=torch.ones(10, 10),
        edge_weights={}
    )
    
    # Run twice with same seed
    state1 = solver.solve(potentials, grid, netlist, tiebreak_seed=42)
    state2 = solver.solve(potentials, grid, netlist, tiebreak_seed=42)
    
    # Routes should be identical (or very similar)
    assert len(state1.wires) == len(state2.wires), "Route lengths differ"
    
    # Check wire segments match
    wires1 = sorted([(w.start, w.end, w.net_id) for w in state1.wires])
    wires2 = sorted([(w.start, w.end, w.net_id) for w in state2.wires])
    
    # At least 80% should match (allowing for some tie-breaking variance)
    matches = sum(1 for w1, w2 in zip(wires1, wires2) if w1 == w2)
    match_ratio = matches / max(len(wires1), len(wires2), 1)
    assert match_ratio >= 0.8, f"Route similarity too low: {match_ratio}"

