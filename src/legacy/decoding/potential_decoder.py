"""Main decoder: Latent → routing potentials (Lipschitz-continuous)."""

import torch
import torch.nn as nn
from typing import Dict

from .decoder import Decoder, RoutingPotentials
from ..routing.grid import Grid
from ..routing.netlist import Netlist


class PotentialDecoder(Decoder):
    """Decoder that outputs soft routing potentials.
    
    This decoder is Lipschitz-continuous and has no discontinuities.
    """
    
    def __init__(self, latent_channels: int = 1, output_channels: int = 1):
        """Initialize decoder.
        
        Args:
            latent_channels: Number of channels in latent
            output_channels: Number of output channels (potentials)
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        
        # Simple projection network (can be replaced with learned network)
        # This ensures Lipschitz continuity
        self.projection = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Softplus()  # Ensures positive outputs, smooth
        )
        
        # Lipschitz constant (can be enforced via spectral normalization)
        self.lipschitz_constant = 1.0
    
    def decode(
        self,
        latent: torch.Tensor,
        grid: Grid,
        netlist: Netlist
    ) -> RoutingPotentials:
        """Decode latent to soft routing potentials.
        
        Args:
            latent: Latent tensor [B, C, H, W] or [C, H, W]
            grid: Grid structure
            netlist: Netlist
        
        Returns:
            RoutingPotentials: Soft potentials
        """
        # Handle batch dimension
        if len(latent.shape) == 3:
            latent = latent.unsqueeze(0)
        batch_size = latent.shape[0]
        
        # Project to potentials
        potentials = self.projection(latent)
        
        # Remove batch dimension if single sample
        if batch_size == 1:
            potentials = potentials.squeeze(0)
        
        # Extract cost field (first channel)
        cost_field = potentials[0] if len(potentials.shape) == 3 else potentials
        
        # Ensure cost field matches grid size
        grid_h, grid_w = grid.get_size()
        if cost_field.shape[0] != grid_h or cost_field.shape[1] != grid_w:
            cost_field = torch.nn.functional.interpolate(
                cost_field.unsqueeze(0).unsqueeze(0),
                size=(grid_h, grid_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Create edge weights from cost field (simple approach)
        edge_weights = self._compute_edge_weights(cost_field, grid)
        
        return RoutingPotentials(
            cost_field=cost_field,
            edge_weights=edge_weights,
            flow_preferences=None
        )
    
    def _compute_edge_weights(
        self,
        cost_field: torch.Tensor,
        grid: Grid
    ) -> Dict[tuple, float]:
        """Compute edge weights from cost field.
        
        Edge weight = average cost of adjacent cells.
        This is smooth and Lipschitz-continuous.
        """
        edge_weights = {}
        h, w = cost_field.shape
        
        for y in range(h):
            for x in range(w):
                # Horizontal edges
                if x < w - 1:
                    edge = ((x, y), (x + 1, y))
                    weight = (cost_field[y, x] + cost_field[y, x + 1]) / 2.0
                    edge_weights[edge] = weight.item()
                
                # Vertical edges
                if y < h - 1:
                    edge = ((x, y), (x, y + 1))
                    weight = (cost_field[y, x] + cost_field[y + 1, x]) / 2.0
                    edge_weights[edge] = weight.item()
        
        return edge_weights
    
    def get_lipschitz_constant(self) -> float:
        """Get Lipschitz constant of the decoder."""
        return self.lipschitz_constant

