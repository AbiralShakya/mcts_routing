"""Diffusion-MCTS Router for nextpnr-xilinx.

Replaces traditional routing with diffusion + MCTS.
Core loop:
1. Rollout: Denoise from noise → routing candidate
2. Prune: Critic predicts bad paths early, kill them
3. Score: Route survivors with real nextpnr router
4. Backprop: UCB updates back to root

Components:
- diffusion/ - Denoising model, netlist-conditioned
- critic/ - GNN predicting routability from partial routing
- mcts/ - Tree, UCB selection, backprop
- bridge/ - Interface to nextpnr router
"""

__version__ = "0.2.0"

# Lazy imports to avoid circular dependencies
# Use explicit imports in user code instead of relying on __init__.py

__all__ = [
    "MCTSRouter",
    "RouterConfig",
    "iterate",
    "RoutingDiffusion",
    "RoutingState",
    "RoutingCritic",
    "NextPNRRouter"
]


def get_mcts_router():
    """Get MCTSRouter class."""
    from .mcts.search import MCTSRouter
    return MCTSRouter


def get_router_config():
    """Get RouterConfig class."""
    from .mcts.search import RouterConfig
    return RouterConfig


def get_routing_diffusion():
    """Get RoutingDiffusion class."""
    from .diffusion.model import RoutingDiffusion
    return RoutingDiffusion


def get_routing_state():
    """Get RoutingState class."""
    from .diffusion.model import RoutingState
    return RoutingState


def get_routing_critic():
    """Get RoutingCritic class."""
    from .critic.gnn import RoutingCritic
    return RoutingCritic


def get_nextpnr_router():
    """Get NextPNRRouter class."""
    from .bridge.router import NextPNRRouter
    return NextPNRRouter

