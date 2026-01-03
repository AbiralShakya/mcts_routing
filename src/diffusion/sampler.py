"""Sampling utilities for routing diffusion."""

import torch
from typing import Optional, Dict, List

from .model import RoutingState, RoutingDiffusion


def denoise_step(
    model: RoutingDiffusion,
    state: RoutingState,
    net_features: torch.Tensor,
    net_positions: torch.Tensor
) -> RoutingState:
    """Single denoising step for routing.

    Commits routing decisions for one or more nets.
    Reduces entropy in the routing latent space.

    Args:
        model: Routing diffusion model
        state: Current routing state
        net_features: Net feature tensor
        net_positions: Net bounding boxes

    Returns:
        New state at timestep t-1
    """
    return model.denoise_step(state, net_features, net_positions)


def sample_routing(
    model: RoutingDiffusion,
    netlist_info: Dict,
    device: str = "cuda"
) -> RoutingState:
    """Sample a complete routing by running full denoising.

    Args:
        model: Routing diffusion model
        netlist_info: Dict with net_features, net_positions, num_pips_per_net
        device: Device to use

    Returns:
        Final routing state at t=0
    """
    net_features = netlist_info["net_features"].to(device)
    net_positions = netlist_info["net_positions"].to(device)
    num_pips_per_net = netlist_info["num_pips_per_net"]

    # Initialize with noise
    net_latents = {}
    for net_id, num_pips in num_pips_per_net.items():
        net_latents[net_id] = torch.randn(num_pips, device=device)

    state = RoutingState(
        net_latents=net_latents,
        timestep=model.num_timesteps,
        routed_nets=set(),
        congestion_map=None
    )

    # Denoise to completion
    while state.timestep > 0:
        state = model.denoise_step(state, net_features, net_positions)

    return state


def initialize_routing_state(
    num_nets: int,
    pips_per_net: List[int],
    num_timesteps: int,
    device: str = "cuda"
) -> RoutingState:
    """Initialize routing state with noise.

    Args:
        num_nets: Number of nets
        pips_per_net: Number of feasible PIPs per net
        num_timesteps: Starting timestep
        device: Device

    Returns:
        Initial noisy routing state
    """
    net_latents = {}
    for i in range(num_nets):
        num_pips = pips_per_net[i] if i < len(pips_per_net) else 100
        net_latents[i] = torch.randn(num_pips, device=device)

    return RoutingState(
        net_latents=net_latents,
        timestep=num_timesteps,
        routed_nets=set(),
        congestion_map=None
    )
