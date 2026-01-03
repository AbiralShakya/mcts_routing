"""Solver interface for converting soft potentials to hard routes."""

from abc import ABC, abstractmethod
from typing import Literal

from ..routing.grid import Grid
from ..routing.netlist import Netlist
from ..routing.state import RoutingState
from ..decoding.decoder import RoutingPotentials


class RoutingSolver(ABC):
    """Base class for routing solvers."""
    
    @abstractmethod
    def solve(
        self,
        potentials: RoutingPotentials,
        grid: Grid,
        netlist: Netlist,
        stability_mode: Literal["randomized_tiebreak", "entropy_regularization", "soft_penalties"] = "randomized_tiebreak",
        k_runs: int = 5
    ) -> RoutingState:
        """
        Convert soft potentials to hard routes.
        
        Args:
            potentials: Soft routing potentials
            grid: Grid structure
            netlist: Netlist
            stability_mode: Stability mechanism to use
            k_runs: Number of runs for randomized tie-breaking
        
        Returns:
            RoutingState: Hard routing solution
        """
        pass

