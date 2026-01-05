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


def iterate(
    root: RoutingNode,
    diffusion: RoutingDiffusion,
    critic: RoutingCritic,
    router: NextPNRRouter,
    grid: Grid,
    netlist: Netlist,
    net_features: torch.Tensor,
    net_positions: torch.Tensor,
    config: RouterConfig
) -> float:
    """Single MCTS iteration.

    This is the core algorithm:
    1. UCB select from root
    2. Denoise until terminal or pruned
    3. Score with real router if terminal
    4. Backpropagate

    Returns:
        Reward achieved (0 if pruned)
    """
    graph_builder = RoutingGraphBuilder(grid)

    # 1. UCB Selection
    node = ucb_select(root, config.ucb_c)

    # 2. Denoise with critic pruning
    while node.t > 0:
        # Denoise step - commit routing decisions
        new_state = diffusion.denoise_step(
            node.state,
            net_features,
            net_positions
        )

        # Create child
        child = RoutingNode(state=new_state, parent=node)
        node.children.append(child)

        # Critic evaluation
        graph = graph_builder.build_graph(new_state, netlist)

        graph = RoutingGraph(
            node_features=graph.node_features.to(config.device),
            edge_index=graph.edge_index.to(config.device),
            edge_features=graph.edge_features.to(config.device),
            congestion=graph.congestion.to(config.device),
            unrouted_mask=graph.unrouted_mask.to(config.device)
        )

        with torch.no_grad():
            # Pass net features and congestion for shared encoders
            critic_score = critic(
                graph,
                net_features=net_features,
                net_positions=net_positions,
                congestion_map=new_state.congestion_map
            ).item()

        if critic_score < config.critic_threshold:
            # PRUNE: Critic predicts routing failure
            child.pruned = True
            backpropagate(child, 0.0)
            return 0.0

        node = child

    # 3. Terminal: Evaluate with REAL router
    # This is the ground truth - no learned surrogate
    routing = diffusion.decode_routing(node.state, {})
    result = router.route_from_assignment(routing, netlist, grid)
    reward = result.as_reward()

    # 4. Backpropagate
    backpropagate(node, reward)

    return reward


class MCTSRouter:
    """MCTS-based router with diffusion and critic.

    Key insight: We optimize for routing success, not wirelength proxy.
    The critic learns what makes routing hard.
    """

    def __init__(
        self,
        diffusion: RoutingDiffusion,
        critic: RoutingCritic,
        router: NextPNRRouter,
        grid: Grid,
        netlist: Netlist,
        config: Optional[RouterConfig] = None,
        device: str = "cuda"
    ):
        self.config = config or RouterConfig()
        self.diffusion = diffusion.to(device)
        self.critic = critic.to(device)
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

        features = torch.zeros(num_nets, 8, device=self.device)
        positions = torch.zeros(num_nets, 4, device=self.device)

        width, height = self.grid.get_size()

        for i, net in enumerate(self.netlist.nets):
            # Fanout
            features[i, 0] = len(net.pins) / 100.0

            # Bounding box
            if net.pins:
                xs = [p.x for p in net.pins]
                ys = [p.y for p in net.pins]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                positions[i, 0] = min_x / width
                positions[i, 1] = min_y / height
                positions[i, 2] = max_x / width
                positions[i, 3] = max_y / height

                # HPWL
                features[i, 1] = ((max_x - min_x) + (max_y - min_y)) / (width + height)

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
