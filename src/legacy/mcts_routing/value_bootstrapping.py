"""Value bootstrapping with proxy rewards for intermediate steps.

R_proxy = -[λ₁·entropy(D(x_t)) + λ₂·unresolved_nets + λ₃·congestion_estimate]
Bootstrap: Q = (1-λ)·R_terminal + λ·R_proxy where λ decays with depth.
"""

import torch
from typing import Optional
import numpy as np

from ..decoding.potential_decoder import PotentialDecoder
from ..decoding.decoder import RoutingPotentials
from ..routing.grid import Grid
from ..routing.netlist import Netlist
from ..routing.state import RoutingState


def compute_proxy_reward(
    latent: torch.Tensor,
    timestep: int,
    grid: Grid,
    netlist: Netlist,
    decoder: Optional[PotentialDecoder] = None,
    routing_state: Optional[RoutingState] = None,
    lambda_entropy: float = 1.0,
    lambda_unresolved: float = 1.0,
    lambda_congestion: float = 1.0
) -> float:
    """Compute cheap proxy reward at intermediate timesteps.
    
    R_proxy = -[λ₁·entropy(D(x_t)) + λ₂·unresolved_nets(x_t) + λ₃·congestion_estimate(x_t)]
    
    Args:
        latent: Latent tensor [C, H, W] or [H, W, C]
        timestep: Current timestep
        grid: Grid structure
        netlist: Netlist
        decoder: Potential decoder
        routing_state: Current routing state (if available)
        lambda_entropy: Weight for entropy term
        lambda_unresolved: Weight for unresolved nets term
        lambda_congestion: Weight for congestion term
    
    Returns:
        Proxy reward value (negative, higher is better)
    """
    if decoder is None:
        # Simple heuristic if no decoder
        return 0.0
    
    # Decode to potentials (cheap operation)
    try:
        potentials = decoder.decode(latent, grid, netlist)
    except Exception:
        return 0.0
    
    # Compute entropy of potential field
    entropy = compute_potential_entropy(potentials)
    
    # Estimate unresolved nets
    unresolved = estimate_unresolved_nets(potentials, netlist, routing_state)
    
    # Estimate congestion
    congestion = estimate_congestion(potentials, grid, routing_state)
    
    # Proxy reward (negative because lower is better)
    proxy_reward = -(
        lambda_entropy * entropy +
        lambda_unresolved * unresolved +
        lambda_congestion * congestion
    )
    
    return float(proxy_reward)


def compute_potential_entropy(potentials: RoutingPotentials) -> float:
    """Compute entropy of potential field.
    
    H(p) = -Σ_{i,j} p(i,j) log p(i,j)
    
    Higher entropy = more uniform = less congestion
    Lower entropy = more concentrated = more congestion
    
    Args:
        potentials: Routing potentials
    
    Returns:
        Entropy value
    """
    cost_field = potentials.cost_field
    
    # Normalize to probability distribution
    cost_sum = cost_field.sum()
    if cost_sum > 1e-8:
        probs = cost_field / cost_sum
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return float(entropy.item())
    return 0.0


def estimate_unresolved_nets(
    potentials: RoutingPotentials,
    netlist: Netlist,
    routing_state: Optional[RoutingState] = None
) -> float:
    """Estimate number of unresolved nets.
    
    Args:
        potentials: Routing potentials
        netlist: Netlist
        routing_state: Current routing state (if available)
    
    Returns:
        Estimated number of unresolved nets
    """
    if routing_state is not None:
        # Count actually unresolved nets
        unresolved = 0
        for net in netlist.nets:
            if not routing_state.is_net_routed(net.net_id):
                unresolved += 1
        return float(unresolved)
    
    # Heuristic: estimate from potentials
    # Check if potential path exists between pins
    disconnected = 0
    cost_field = potentials.cost_field
    
    for net in netlist.nets:
        if len(net.pins) >= 2:
            # Check if cost at pins is reasonable
            p1 = net.pins[0]
            p2 = net.pins[1]
            if (0 <= p1.y < cost_field.shape[0] and 0 <= p1.x < cost_field.shape[1] and
                0 <= p2.y < cost_field.shape[0] and 0 <= p2.x < cost_field.shape[1]):
                cost1 = cost_field[p1.y, p1.x].item()
                cost2 = cost_field[p2.y, p2.x].item()
                # High cost threshold suggests disconnection
                if cost1 > 0.9 or cost2 > 0.9:
                    disconnected += 1
    
    return float(disconnected)


def estimate_congestion(
    potentials: RoutingPotentials,
    grid: Grid,
    routing_state: Optional[RoutingState] = None
) -> float:
    """Estimate congestion from potentials.
    
    congestion_estimate = Σ_{cell} max(0, predicted_usage(cell) - capacity)²
    
    Args:
        potentials: Routing potentials
        grid: Grid structure
        routing_state: Current routing state (if available)
    
    Returns:
        Congestion estimate
    """
    cost_field = potentials.cost_field
    H, W = cost_field.shape
    
    # Estimate usage from cost field (high cost = high usage)
    # Assume capacity is uniform (can be made more sophisticated)
    capacity = 1.0  # Normalized capacity
    
    congestion_sum = 0.0
    for y in range(H):
        for x in range(W):
            usage = cost_field[y, x].item()
            congestion = max(0.0, usage - capacity) ** 2
            congestion_sum += congestion
    
    return congestion_sum


def bootstrap_q_value(
    terminal_reward: float,
    proxy_reward: float,
    timestep: int,
    T: int,
    lambda_decay: float = 0.5
) -> float:
    """Compute bootstrapped Q-value.
    
    Q(s_t, a) = (1-λ)·R_terminal + λ·R_proxy(s_{t-1})
    where λ decays with depth: λ = lambda_decay * (1 - t/T)
    
    Args:
        terminal_reward: Terminal reward R(x_0)
        proxy_reward: Proxy reward R_proxy(x_t)
        timestep: Current timestep t
        T: Total timesteps
        lambda_decay: Base decay factor
    
    Returns:
        Bootstrapped Q-value
    """
    # Lambda decays with depth
    lambda_t = lambda_decay * (1.0 - timestep / T)
    
    # Bootstrap target
    q_value = (1.0 - lambda_t) * terminal_reward + lambda_t * proxy_reward
    
    return q_value


def compute_congestion_entropy(potentials: RoutingPotentials) -> float:
    """Compute congestion entropy from potentials (legacy function for compatibility).
    
    Args:
        potentials: Routing potentials
    
    Returns:
        Entropy value
    """
    return compute_potential_entropy(potentials)


def estimate_disconnected_nets(
    potentials: RoutingPotentials,
    netlist: Netlist
) -> float:
    """Estimate number of disconnected nets (legacy function for compatibility).
    
    Args:
        potentials: Routing potentials
        netlist: Netlist
    
    Returns:
        Estimated disconnected nets
    """
    return estimate_unresolved_nets(potentials, netlist, routing_state=None)
