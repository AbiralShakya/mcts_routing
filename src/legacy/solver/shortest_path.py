"""Shortest-path solver (Dijkstra/A*) on potential-weighted graph."""

import heapq
from typing import List, Tuple, Optional, Dict, Set
import random

from .solver_interface import RoutingSolver
from ..routing.grid import Grid
from ..routing.netlist import Netlist, Net, Pin
from ..routing.state import RoutingState, WireSegment
from ..decoding.decoder import RoutingPotentials


class ShortestPathSolver(RoutingSolver):
    """Shortest-path solver using Dijkstra's algorithm."""
    
    def solve(
        self,
        potentials: RoutingPotentials,
        grid: Grid,
        netlist: Netlist,
        stability_mode: str = "randomized_tiebreak",
        k_runs: int = 5,
        tiebreak_seed: Optional[int] = None
    ) -> RoutingState:
        """Solve routing using shortest-path algorithm.
        
        Args:
            potentials: Soft routing potentials
            grid: Grid structure
            netlist: Netlist
            stability_mode: Stability mechanism
            k_runs: Number of runs for averaging
            tiebreak_seed: Random seed for tie-breaking
        
        Returns:
            RoutingState: Hard routing solution
        """
        # Apply stability mechanism
        if stability_mode == "entropy_regularization":
            from .stability import add_entropy_regularization
            potentials = add_entropy_regularization(potentials)
        
        # Create routing state
        state = RoutingState(grid=grid, netlist=netlist)
        
        # Route each net
        for net in netlist.nets:
            route = self._route_net(net, potentials, grid, tiebreak_seed)
            if route:
                # Add wires for the route
                for i in range(len(route) - 1):
                    wire = WireSegment(
                        start=route[i],
                        end=route[i + 1],
                        net_id=net.net_id,
                        layer=0
                    )
                    state.add_wire(wire)
        
        return state
    
    def _route_net(
        self,
        net: Net,
        potentials: RoutingPotentials,
        grid: Grid,
        tiebreak_seed: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Route a single net using Dijkstra's algorithm.
        
        Args:
            net: Net to route
            potentials: Routing potentials
            grid: Grid structure
            tiebreak_seed: Random seed for tie-breaking
        
        Returns:
            List of (x, y) coordinates representing the route
        """
        if tiebreak_seed is not None:
            random.seed(tiebreak_seed)
        
        # Get source and sinks
        source = net.source
        sinks = net.sinks
        
        # Use Dijkstra to find shortest path from source to each sink
        all_routes = []
        
        current = source
        remaining_sinks = set(sinks)
        
        while remaining_sinks:
            # Find shortest path to nearest sink
            best_route = None
            best_sink = None
            best_cost = float('inf')
            
            for sink in remaining_sinks:
                route, cost = self._dijkstra(
                    (current.x, current.y),
                    (sink.x, sink.y),
                    potentials,
                    grid
                )
                if route and cost < best_cost:
                    best_route = route
                    best_sink = sink
                    best_cost = cost
            
            if best_route:
                all_routes.extend(best_route[1:])  # Skip first point (already in route)
                current = best_sink
                remaining_sinks.remove(best_sink)
            else:
                break  # Cannot route to remaining sinks
        
        return [(source.x, source.y)] + all_routes if all_routes else []
    
    def _dijkstra(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        potentials: RoutingPotentials,
        grid: Grid
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Dijkstra's algorithm with potential-weighted edges.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            potentials: Routing potentials
            grid: Grid structure
        
        Returns:
            (route, cost): Route as list of positions and total cost
        """
        # Priority queue: (cost, position, path)
        pq = [(0.0, start, [start])]
        visited: Set[Tuple[int, int]] = set()
        
        while pq:
            cost, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return path, cost
            
            # Explore neighbors
            x, y = current
            for nx, ny in grid.get_neighbors(x, y):
                if (nx, ny) in visited:
                    continue
                
                # Get edge weight from potentials
                edge = (current, (nx, ny))
                if edge in potentials.edge_weights:
                    edge_cost = potentials.edge_weights[edge]
                else:
                    # Fallback: use average of cell costs
                    if y < potentials.cost_field.shape[0] and x < potentials.cost_field.shape[1]:
                        cell_cost = potentials.cost_field[y, x].item()
                    else:
                        cell_cost = 1.0
                    edge_cost = cell_cost
                
                new_cost = cost + edge_cost
                new_path = path + [(nx, ny)]
                
                heapq.heappush(pq, (new_cost, (nx, ny), new_path))
        
        return [], float('inf')  # No path found

