"""Min-cost flow solver (placeholder - can be extended)."""

from .solver_interface import RoutingSolver
from ..routing.grid import Grid
from ..routing.netlist import Netlist
from ..routing.state import RoutingState
from ..decoding.decoder import RoutingPotentials
from .shortest_path import ShortestPathSolver


class MinCostFlowSolver(RoutingSolver):
    """Min-cost flow solver (currently uses shortest-path as fallback)."""
    
    def __init__(self):
        self.fallback_solver = ShortestPathSolver()
    
    def solve(
        self,
        potentials: RoutingPotentials,
        grid: Grid,
        netlist: Netlist,
        stability_mode: str = "randomized_tiebreak",
        k_runs: int = 5
    ) -> RoutingState:
        """Solve using min-cost flow (fallback to shortest-path for now)."""
        # TODO: Implement proper min-cost flow algorithm
        # For now, use shortest-path as fallback
        return self.fallback_solver.solve(potentials, grid, netlist, stability_mode, k_runs)

