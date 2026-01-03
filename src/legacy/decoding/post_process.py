"""Post-processing for decoded potentials (smoothing, regularization)."""

import torch
from typing import Optional

from .decoder import RoutingPotentials


def smooth_potentials(
    potentials: RoutingPotentials,
    kernel_size: int = 3,
    sigma: float = 1.0
) -> RoutingPotentials:
    """Smooth potentials using Gaussian blur.
    
    This maintains Lipschitz continuity while reducing noise.
    """
    cost_field = potentials.cost_field
    
    # Apply Gaussian blur
    if kernel_size > 1:
        # Simple box filter approximation (can use proper Gaussian)
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        cost_field = cost_field.unsqueeze(0).unsqueeze(0)
        cost_field = torch.nn.functional.conv2d(
            cost_field,
            kernel.to(cost_field.device),
            padding=kernel_size // 2
        ).squeeze(0).squeeze(0)
    
    return RoutingPotentials(
        cost_field=cost_field,
        edge_weights=potentials.edge_weights,
        flow_preferences=potentials.flow_preferences
    )


def normalize_potentials(
    potentials: RoutingPotentials,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> RoutingPotentials:
    """Normalize potentials to [min_val, max_val] range.
    
    This maintains Lipschitz continuity.
    """
    cost_field = potentials.cost_field
    
    # Normalize
    min_cost = cost_field.min()
    max_cost = cost_field.max()
    if max_cost > min_cost:
        cost_field = (cost_field - min_cost) / (max_cost - min_cost)
        cost_field = cost_field * (max_val - min_val) + min_val
    
    return RoutingPotentials(
        cost_field=cost_field,
        edge_weights=potentials.edge_weights,
        flow_preferences=potentials.flow_preferences
    )

