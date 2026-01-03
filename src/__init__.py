"""Diffusion-MCTS Placer for nextpnr-xilinx.

Replaces simulated annealing with diffusion + MCTS.
Core loop:
1. Rollout: Denoise from noise → placement
2. Prune: Critic predicts bad paths early, kill them
3. Score: Route survivors with real nextpnr router
4. Backprop: UCB updates back to root

Components:
- diffusion/ - Denoising model, netlist-conditioned
- critic/ - GNN predicting routability from partial placement
- mcts/ - Tree, UCB selection, backprop
- bridge/ - C++ bindings to nextpnr router
"""

__version__ = "0.2.0"

from .mcts.search import MCTSPlacer, PlacerConfig, iterate
from .diffusion.model import PlacementDiffusion, PlacementState
from .critic.gnn import RoutabilityCritic
from .bridge.router import NextPNRRouter

__all__ = [
    "MCTSPlacer",
    "PlacerConfig",
    "iterate",
    "PlacementDiffusion",
    "PlacementState",
    "RoutabilityCritic",
    "NextPNRRouter"
]


def run_placer(design_path: str, output_path: str, **kwargs):
    """High-level API to run the placer.

    Args:
        design_path: Path to nextpnr design JSON
        output_path: Path to write placed design
        **kwargs: Placer configuration options

    Returns:
        Placement result
    """
    from .integration.nextpnr.reader import NextPNRReader
    from .bridge.placement_io import export_placement

    config = PlacerConfig(**kwargs)
    reader = NextPNRReader()

    # Read design
    grid, netlist, _, _ = reader.read_all(design_path)

    # Create components
    diffusion = PlacementDiffusion(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    critic = RoutabilityCritic(
        hidden_dim=config.critic_hidden_dim,
        num_layers=config.critic_num_layers
    )
    router = NextPNRRouter(
        nextpnr_path=config.nextpnr_path,
        chipdb_path=config.chipdb_path,
        timeout_seconds=config.router_timeout
    )

    # Create placer and run
    placer = MCTSPlacer(
        diffusion=diffusion,
        critic=critic,
        router=router,
        grid=grid,
        netlist=netlist,
        ucb_c=config.ucb_c,
        critic_threshold=config.critic_threshold,
        max_simulations=config.max_simulations,
        device=config.device
    )

    placement = placer.search()

    # Write output
    export_placement(placement, netlist, grid, output_path)

    return placement

