"""Decoder interface: latent → soft routing potentials."""

from abc import ABC, abstractmethod
from typing import Dict
import torch

from ..routing.grid import Grid
from ..routing.netlist import Netlist


class RoutingPotentials:
    """Soft routing potentials (cost fields, edge weights).
    
    These are continuous-valued and Lipschitz-continuous w.r.t. latent.
    """
    
    def __init__(
        self,
        cost_field: torch.Tensor,  # [H, W] - cost at each cell
        edge_weights: Dict[tuple, float] = None,  # Edge weights
        flow_preferences: torch.Tensor = None  # [H, W] - flow direction preferences
    ):
        self.cost_field = cost_field
        self.edge_weights = edge_weights or {}
        self.flow_preferences = flow_preferences
    
    def __repr__(self) -> str:
        return f"RoutingPotentials(shape={self.cost_field.shape})"


class Decoder(ABC):
    """Base decoder interface."""
    
    @abstractmethod
    def decode(
        self,
        latent: torch.Tensor,
        grid: Grid,
        netlist: Netlist
    ) -> RoutingPotentials:
        """
        Decode latent to soft routing potentials.
        
        Args:
            latent: Latent tensor [B, C, H, W] or [C, H, W]
            grid: Grid structure
            netlist: Netlist
        
        Returns:
            RoutingPotentials: Soft potentials (NOT hard routes)
        
        Properties:
            - Lipschitz-continuous: ||decode(x) - decode(x')|| ≤ L ||x - x'||
            - No discontinuities (no thresholds, no argmax)
            - Continuous-valued
        """
        pass

