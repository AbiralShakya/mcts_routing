"""Main MCTS search for routing.

The algorithm from the spec:

def iterate(root):
    node = ucb_select(root)
    while node.t > 0:
        node = denoise_step(node)
        if critic(node) < threshold:
            return backprop(node, 0)  # pruned
    score = nextpnr_route(node.routing)
    backprop(node, score)

This is AlphaZero-style planning for FPGA routing:
- Diffusion supplies a structured policy over an enormous discrete graph
- A critic prunes impossible futures early
- The real router provides exact terminal rewards
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .node import RoutingNode
from .tree import MCTSTree
from .ucb import ucb_select
from .backprop import backpropagate

from ..diffusion.model import RoutingDiffusion, RoutingState
from ..diffusion.sampler import initialize_routing_state
from ..critic.gnn import RoutingCritic, RoutingGraph
from ..critic.features import RoutingGraphBuilder
from ..bridge.router import NextPNRRouter
from ..core.routing.netlist import Netlist
from ..core.routing.grid import Grid
from typing import Optional


@dataclass
class RouterConfig:
    """Configuration for MCTS router."""
    num_timesteps: int = 1000
    hidden_dim: int = 256
    num_layers: int = 6
    max_pips_per_net: int = 1000
    ucb_c: float = 1.41
    max_iterations: int = 1000
    critic_threshold: float = 0.3
    critic_hidden_dim: int = 128
    critic_num_layers: int = 4
    nextpnr_path: str = "nextpnr-xilinx"
    chipdb_path: Optional[str] = None
    router_timeout: float = 60.0
    device: str = "cuda"
    # MCTS branching: number of children to expand per node
    # Setting to 1 is rejection sampling (old behavior)
    # Setting to 4+ enables true MCTS tree exploration
    num_branches: int = 4
    # Noise scale for stochastic branching
    branch_noise_scale: float = 0.1


def iterate(
    root: RoutingNode,
    diffusion: RoutingDiffusion,
    critic: Optional[RoutingCritic],
    router: NextPNRRouter,
    grid: Grid,
    netlist: Netlist,
    net_features: torch.Tensor,
    net_positions: torch.Tensor,
    config: RouterConfig
) -> float:
    """Single MCTS iteration with K-branch expansion.

    This implements TRUE MCTS with multiple branches per node:
    1. UCB select from root
    2. Expand K children with different noise samples
    3. Evaluate all children with time-aware critic
    4. Prune children below threshold
    5. UCB select from non-pruned children
    6. Continue denoising until terminal or all pruned
    7. Score with real router if terminal
    8. Backpropagate

    The key difference from rejection sampling (1 child):
    - Multiple branches explore alternative denoising paths
    - Critic pruning removes bad branches early
    - UCB selection balances exploration vs exploitation

    Returns:
        Reward achieved (0 if all branches pruned)
    """
    graph_builder = RoutingGraphBuilder(grid)

    # 1. UCB Selection from root
    node = ucb_select(root, config.ucb_c)

    # 2. Denoise with K-branch expansion and critic pruning
    while node.state.timestep > 0:
        # Get latent shape for generating noise
        num_nets = len(node.state.net_latents)
        latent_shape = diffusion.get_latent_shape(num_nets)

        # Expand K children with different noise samples
        for k in range(config.num_branches):
            # Sample stochastic noise for this branch
            noise_k = torch.randn(latent_shape, device=config.device)

            # Denoise with this specific noise
            new_state = diffusion.denoise_step_stochastic(
                node.state,
                net_features,
                net_positions,
                noise=noise_k,
                noise_scale=config.branch_noise_scale
            )

            # Create child node
            child = RoutingNode(state=new_state, parent=node)
            node.children.append(child)

            # Critic evaluation with time-aware scoring
            if critic is not None:
                graph = graph_builder.build_graph(new_state, netlist)

                graph = RoutingGraph(
                    node_features=graph.node_features.to(config.device),
                    edge_index=graph.edge_index.to(config.device),
                    edge_features=graph.edge_features.to(config.device),
                    congestion=graph.congestion.to(config.device),
                    unrouted_mask=graph.unrouted_mask.to(config.device)
                )

                with torch.no_grad():
                    # Pass timestep for time-aware evaluation
                    # Critical: at high t, noisy states are expected and should not be penalized
                    timestep_tensor = torch.tensor([new_state.timestep], device=config.device)

                    critic_score = critic(
                        graph,
                        timestep=timestep_tensor,
                        net_features=net_features,
                        net_positions=net_positions,
                        congestion_map=new_state.congestion_map
                    ).item()

                # Store critic score on child for UCB selection
                child.critic_score = critic_score

                if critic_score < config.critic_threshold:
                    # Mark as pruned but don't backpropagate yet
                    child.pruned = True

        # 3. UCB select from non-pruned children
        non_pruned = [c for c in node.children if not c.pruned]

        if not non_pruned:
            # All children pruned - backpropagate failure
            # Pick any child to backpropagate through
            if node.children:
                backpropagate(node.children[0], 0.0)
            return 0.0

        # Select best child using UCB (considering critic scores as prior)
        node = _ucb_select_from_list(non_pruned, config.ucb_c)

    # 4. Terminal: Evaluate with REAL router
    # This is the ground truth - no learned surrogate
    routing = diffusion.decode_routing(node.state, {})
    result = router.route_from_assignment(routing, netlist, grid)
    reward = result.as_reward()

    # 5. Backpropagate
    backpropagate(node, reward)

    return reward


def _ucb_select_from_list(children: list, ucb_c: float) -> RoutingNode:
    """Select best child from list using UCB formula.

    Args:
        children: List of non-pruned child nodes
        ucb_c: UCB exploration constant

    Returns:
        Selected child node
    """
    import math

    if not children:
        raise ValueError("No children to select from")

    if len(children) == 1:
        return children[0]

    # Compute parent visit count
    parent = children[0].parent
    parent_visits = parent.visit_count if parent else 1

    best_child = None
    best_ucb = float('-inf')

    for child in children:
        if child.visit_count == 0:
            # Unvisited nodes get priority (infinite UCB)
            # But use critic score to break ties among unvisited
            ucb = float('inf')
            if child.critic_score is not None:
                ucb = 1e9 + child.critic_score  # High base + critic tiebreaker
        else:
            # Standard UCB formula
            exploitation = child.Q  # Use Q property (average value)
            exploration = ucb_c * math.sqrt(math.log(parent_visits + 1) / child.visit_count)
            ucb = exploitation + exploration

        if ucb > best_ucb:
            best_ucb = ucb
            best_child = child

    return best_child


class MCTSRouter:
    """MCTS-based router with diffusion and critic.

    Key insight: We optimize for routing success, not wirelength proxy.
    The critic learns what makes routing hard.
    """

    def __init__(
        self,
        diffusion: RoutingDiffusion,
        critic: Optional[RoutingCritic],
        router: NextPNRRouter,
        grid: Grid,
        netlist: Netlist,
        config: Optional[RouterConfig] = None,
        device: str = "cuda"
    ):
        self.config = config or RouterConfig()
        self.diffusion = diffusion.to(device)
        self.critic = critic.to(device) if critic is not None else None
        self.router = router
        self.grid = grid
        self.netlist = netlist
        self.device = device

        # Precompute net features
        self.net_features, self.net_positions = self._compute_net_features()

        # Statistics
        self.num_pruned = 0
        self.num_routed = 0

    def _compute_net_features(self):
        """Compute net feature tensors."""
        num_nets = len(self.netlist.nets)

        features = torch.zeros(num_nets, 7, device=self.device)  # Changed from 8 to 7
        positions = torch.zeros(num_nets, 4, device=self.device)

        width, height = self.grid.get_size()

        for i, net in enumerate(self.netlist.nets):
            if net.pins:
                xs = [p.x for p in net.pins]
                ys = [p.y for p in net.pins]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                bbox_width = max_x - min_x
                bbox_height = max_y - min_y
                hpwl = bbox_width + bbox_height
                
                # Compute 7 features matching training data format
                features[i, 0] = (len(net.pins) - 1) / 10.0  # normalized fanout
                features[i, 1] = bbox_width / max(width, 1)  # normalized bbox width
                features[i, 2] = bbox_height / max(height, 1)  # normalized bbox height
                features[i, 3] = hpwl / (width + height)  # normalized HPWL
                features[i, 4] = (min_x + max_x) / 2.0 / max(width, 1)  # center x
                features[i, 5] = (min_y + max_y) / 2.0 / max(height, 1)  # center y
                features[i, 6] = len(net.pins) / 10.0  # normalized pin count

                # Positions (bounding box)
                positions[i, 0] = min_x / max(width, 1)
                positions[i, 1] = min_y / max(height, 1)
                positions[i, 2] = max_x / max(width, 1)
                positions[i, 3] = max_y / max(height, 1)

        return features, positions

    def route(self, max_iterations: Optional[int] = None) -> Dict[int, list]:
        """Run MCTS routing and return best result.

        Returns:
            Dict[net_id -> list of PIPs]
        """
        max_iter = max_iterations or self.config.max_iterations

        # Initialize root with noise (fully unassigned routing)
        pips_per_net = [100] * len(self.netlist.nets)  # Simplified
        initial_state = initialize_routing_state(
            len(self.netlist.nets),
            pips_per_net,
            self.diffusion.num_timesteps,
            self.device
        )
        root = RoutingNode(state=initial_state)

        # Run iterations
        for i in range(max_iter):
            reward = iterate(
                root=root,
                diffusion=self.diffusion,
                critic=self.critic,
                router=self.router,
                grid=self.grid,
                netlist=self.netlist,
                net_features=self.net_features,
                net_positions=self.net_positions,
                config=self.config
            )

            if reward == 0.0:
                self.num_pruned += 1
            else:
                self.num_routed += 1

            if (i + 1) % 100 == 0:
                tree = MCTSTree(root)
                stats = tree.get_statistics()
                print(f"Iteration {i + 1}/{max_iter}: "
                      f"pruned={self.num_pruned}, routed={self.num_routed}, "
                      f"nodes={stats['total_nodes']}, Q={stats['root_Q']:.3f}")

        # Return best routing
        tree = MCTSTree(root)
        best_node = tree.get_best_terminal()

        if best_node is None:
            return {}

        return self.diffusion.decode_routing(best_node.state, {})

    def get_statistics(self) -> dict:
        """Return search statistics."""
        return {
            "num_pruned": self.num_pruned,
            "num_routed": self.num_routed,
            "prune_rate": self.num_pruned / (self.num_pruned + self.num_routed + 1e-8)
        }
