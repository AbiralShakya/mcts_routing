#!/usr/bin/env python3
"""Run MCTS inference with trained diffusion and critic models.

This implements the full MCTS routing pipeline:
1. Load trained diffusion model (policy)
2. Load trained critic model (value)
3. Run MCTS search with both models
4. Optionally verify with nextpnr

Usage:
    python experiments/run_mcts.py \
        --joint_checkpoint checkpoints/joint_model.pt \
        --data_dir data/routing_states \
        --num_samples 10 \
        --num_iterations 500
"""

import argparse
import yaml
import torch
import pickle
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.encoders import create_shared_encoders
from src.diffusion.model import create_routing_diffusion, RoutingState
from src.diffusion.sampler import initialize_routing_state
from src.critic.gnn import RoutingCritic, RoutingGraph
from src.critic.features import RoutingGraphBuilder
from src.mcts.node import RoutingNode
from src.mcts.tree import MCTSTree
from src.mcts.ucb import ucb_select
from src.mcts.backprop import backpropagate
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCTSRouter:
    """MCTS Router using trained diffusion and critic models."""

    def __init__(
        self,
        diffusion,
        critic,
        device: str = "cuda",
        ucb_c: float = 1.41,
        critic_threshold: float = 0.3,
        max_depth: int = 100
    ):
        self.diffusion = diffusion.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.ucb_c = ucb_c
        self.critic_threshold = critic_threshold
        self.max_depth = max_depth

        self.diffusion.eval()
        self.critic.eval()

        # Stats
        self.num_iterations = 0
        self.num_pruned = 0
        self.num_terminal = 0

    def create_initial_state(
        self,
        num_nets: int,
        pips_per_net: list,
        num_timesteps: int
    ) -> RoutingState:
        """Create initial noisy routing state."""
        return initialize_routing_state(
            num_nets=num_nets,
            pips_per_net=pips_per_net,
            num_timesteps=num_timesteps,
            device=self.device
        )

    def evaluate_with_critic(
        self,
        state: RoutingState,
        graph_builder: RoutingGraphBuilder,
        netlist: Netlist,
        net_features: torch.Tensor,
        net_positions: torch.Tensor
    ) -> float:
        """Evaluate state with critic model."""
        # Build graph representation
        graph = graph_builder.build_graph(state, netlist)

        # Move to device
        graph = RoutingGraph(
            node_features=graph.node_features.to(self.device),
            edge_index=graph.edge_index.to(self.device),
            edge_features=graph.edge_features.to(self.device),
            congestion=graph.congestion.to(self.device),
            unrouted_mask=graph.unrouted_mask.to(self.device)
        )

        with torch.no_grad():
            score = self.critic(
                graph,
                net_features=net_features,
                net_positions=net_positions,
                congestion_map=state.congestion_map
            )

        return score.item()

    def mcts_iterate(
        self,
        root: RoutingNode,
        net_features: torch.Tensor,
        net_positions: torch.Tensor,
        graph_builder: RoutingGraphBuilder,
        netlist: Netlist
    ) -> float:
        """Single MCTS iteration."""
        # 1. UCB Selection
        node = ucb_select(root, self.ucb_c)

        # 2. Denoise with critic pruning
        depth = 0
        while node.state.timestep > 0 and depth < self.max_depth:
            # Denoise step
            with torch.no_grad():
                new_state = self.diffusion.denoise_step(
                    node.state,
                    net_features,
                    net_positions
                )

            # Create child node
            child = RoutingNode(state=new_state, parent=node)
            node.children.append(child)

            # Evaluate with critic
            critic_score = self.evaluate_with_critic(
                new_state, graph_builder, netlist, net_features, net_positions
            )

            # Pruning check
            if critic_score < self.critic_threshold:
                child.pruned = True
                backpropagate(child, 0.0)
                self.num_pruned += 1
                return 0.0

            node = child
            depth += 1

        # 3. Terminal state reached
        self.num_terminal += 1

        # Compute final reward based on routing quality
        # In production, this would call nextpnr
        # For now, use critic score as proxy
        final_score = self.evaluate_with_critic(
            node.state, graph_builder, netlist, net_features, net_positions
        )

        # 4. Backpropagate
        backpropagate(node, final_score)

        return final_score

    def search(
        self,
        netlist: Netlist,
        grid: Grid,
        net_features: torch.Tensor,
        net_positions: torch.Tensor,
        num_iterations: int = 500,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run MCTS search for routing."""
        start_time = time.time()

        # Create graph builder
        graph_builder = RoutingGraphBuilder(grid)

        # Estimate PIPs per net
        pips_per_net = []
        for net in netlist.nets:
            if len(net.pins) >= 2:
                xs = [p.x for p in net.pins]
                ys = [p.y for p in net.pins]
                hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                estimated_pips = max(10, hpwl * 2 + len(net.pins) * 3)
            else:
                estimated_pips = 10
            pips_per_net.append(int(estimated_pips))

        # Initialize root
        initial_state = self.create_initial_state(
            num_nets=len(netlist.nets),
            pips_per_net=pips_per_net,
            num_timesteps=self.diffusion.num_timesteps
        )
        root = RoutingNode(state=initial_state)

        # Run iterations
        rewards = []
        for i in range(num_iterations):
            reward = self.mcts_iterate(
                root, net_features, net_positions, graph_builder, netlist
            )
            rewards.append(reward)
            self.num_iterations += 1

            if verbose and (i + 1) % 100 == 0:
                tree = MCTSTree(root)
                stats = tree.get_statistics()
                avg_reward = sum(rewards[-100:]) / len(rewards[-100:])
                logger.info(
                    f"Iter {i+1}/{num_iterations}: "
                    f"avg_reward={avg_reward:.3f}, "
                    f"pruned={self.num_pruned}, "
                    f"terminal={self.num_terminal}, "
                    f"nodes={stats['total_nodes']}"
                )

        elapsed = time.time() - start_time

        # Get best routing
        tree = MCTSTree(root)
        best_node = tree.get_best_terminal()

        if best_node is not None:
            best_routing = self.diffusion.decode_routing(best_node.state, {})
            best_score = best_node.q_value
        else:
            best_routing = {}
            best_score = 0.0

        return {
            'best_routing': best_routing,
            'best_score': best_score,
            'num_iterations': num_iterations,
            'num_pruned': self.num_pruned,
            'num_terminal': self.num_terminal,
            'elapsed_time': elapsed,
            'tree_stats': tree.get_statistics(),
            'rewards': rewards
        }


def load_joint_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> tuple:
    """Load joint checkpoint with shared encoders."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    hidden_dim = model_config.get('hidden_dim', 256)
    net_feat_dim = model_config.get('net_feat_dim', 7)

    # Create shared encoders
    net_encoder, cong_encoder = create_shared_encoders(
        hidden_dim=hidden_dim,
        net_feat_dim=net_feat_dim
    )

    # Load shared encoder weights if available
    if 'shared_net_encoder_state_dict' in checkpoint:
        net_encoder.load_state_dict(checkpoint['shared_net_encoder_state_dict'])
    if 'shared_congestion_encoder_state_dict' in checkpoint:
        cong_encoder.load_state_dict(checkpoint['shared_congestion_encoder_state_dict'])

    # Create diffusion
    diffusion = create_routing_diffusion(
        config=model_config,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder
    )
    if 'diffusion_state_dict' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'])

    # Create critic
    critic = RoutingCritic(
        node_dim=model_config.get('node_dim', 64),
        edge_dim=model_config.get('edge_dim', 32),
        hidden_dim=hidden_dim,
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder,
        net_feat_dim=net_feat_dim
    )
    if 'critic_state_dict' in checkpoint:
        critic.load_state_dict(checkpoint['critic_state_dict'])

    return diffusion, critic, config


def load_sample(data_dir: str, idx: int = 0) -> dict:
    """Load a sample from the dataset."""
    data_path = Path(data_dir)
    pkl_files = sorted(data_path.glob("*.pkl"))

    if not pkl_files:
        raise ValueError(f"No .pkl files found in {data_dir}")

    sample_file = pkl_files[idx % len(pkl_files)]
    with open(sample_file, 'rb') as f:
        sample = pickle.load(f)

    return sample


def main():
    parser = argparse.ArgumentParser(description="Run MCTS routing inference")
    parser.add_argument("--joint_checkpoint", type=str, default=None,
                        help="Path to joint checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, default=None,
                        help="Path to diffusion checkpoint")
    parser.add_argument("--critic_checkpoint", type=str, default=None,
                        help="Path to critic checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file (if not using joint checkpoint)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory with routing samples")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to run")
    parser.add_argument("--num_iterations", type=int, default=500,
                        help="MCTS iterations per sample")
    parser.add_argument("--ucb_c", type=float, default=1.41,
                        help="UCB exploration constant")
    parser.add_argument("--critic_threshold", type=float, default=0.3,
                        help="Critic pruning threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Load models
    if args.joint_checkpoint:
        logger.info(f"Loading joint checkpoint from {args.joint_checkpoint}")
        diffusion, critic, config = load_joint_checkpoint(args.joint_checkpoint, args.device)
    else:
        # Load separately
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        model_config = config.get('model', {})
        hidden_dim = model_config.get('hidden_dim', 256)
        net_feat_dim = model_config.get('net_feat_dim', 7)

        net_encoder, cong_encoder = create_shared_encoders(hidden_dim, net_feat_dim)

        diffusion = create_routing_diffusion(
            config=model_config,
            shared_net_encoder=net_encoder,
            shared_congestion_encoder=cong_encoder
        )

        if args.diffusion_checkpoint:
            ckpt = torch.load(args.diffusion_checkpoint, map_location=args.device)
            diffusion.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Loaded diffusion from {args.diffusion_checkpoint}")

        critic = RoutingCritic(
            hidden_dim=hidden_dim,
            shared_net_encoder=net_encoder,
            shared_congestion_encoder=cong_encoder,
            net_feat_dim=net_feat_dim
        )

        if args.critic_checkpoint:
            ckpt = torch.load(args.critic_checkpoint, map_location=args.device)
            critic.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Loaded critic from {args.critic_checkpoint}")

    # Create router
    router = MCTSRouter(
        diffusion=diffusion,
        critic=critic,
        device=args.device,
        ucb_c=args.ucb_c,
        critic_threshold=args.critic_threshold
    )

    logger.info("="*60)
    logger.info("MCTS Routing Inference")
    logger.info("="*60)

    # Run on samples
    all_results = []

    for i in range(args.num_samples):
        logger.info(f"\nSample {i+1}/{args.num_samples}")
        logger.info("-"*40)

        # Load sample
        sample = load_sample(args.data_dir, i)

        # Extract netlist info
        routing_state = sample['routing_state']
        net_features = torch.tensor(sample['net_features'], device=args.device).float()
        net_positions = torch.tensor(sample['net_positions'], device=args.device).float()

        if net_features.dim() == 2:
            net_features = net_features.unsqueeze(0)
            net_positions = net_positions.unsqueeze(0)

        # Create netlist and grid from sample
        num_nets = len(routing_state.net_latents)
        grid_size = sample.get('grid_size', 20)

        from src.core.routing.netlist import Netlist, Net, Pin
        from src.core.routing.grid import Grid

        # Create dummy netlist (in production, load actual netlist)
        nets = []
        for net_id in range(num_nets):
            pins = [Pin(x=0, y=0, pin_id=0), Pin(x=grid_size//2, y=grid_size//2, pin_id=1)]
            nets.append(Net(net_id=net_id, pins=pins, name=f"net_{net_id}"))

        netlist = Netlist(nets=nets)
        grid = Grid(width=grid_size, height=grid_size)

        # Reset router stats
        router.num_iterations = 0
        router.num_pruned = 0
        router.num_terminal = 0

        # Run MCTS
        result = router.search(
            netlist=netlist,
            grid=grid,
            net_features=net_features,
            net_positions=net_positions,
            num_iterations=args.num_iterations
        )

        all_results.append(result)

        logger.info(f"Best score: {result['best_score']:.4f}")
        logger.info(f"Pruned: {result['num_pruned']}, Terminal: {result['num_terminal']}")
        logger.info(f"Time: {result['elapsed_time']:.2f}s")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    avg_score = sum(r['best_score'] for r in all_results) / len(all_results)
    avg_pruned = sum(r['num_pruned'] for r in all_results) / len(all_results)
    avg_time = sum(r['elapsed_time'] for r in all_results) / len(all_results)

    logger.info(f"Average best score: {avg_score:.4f}")
    logger.info(f"Average pruned: {avg_pruned:.1f}")
    logger.info(f"Average time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
