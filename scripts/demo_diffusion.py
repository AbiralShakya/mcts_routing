#!/usr/bin/env python3
"""End-to-end demo of diffusion-based routing.

This script demonstrates:
1. Creating synthetic netlists and grids
2. Initializing the diffusion model
3. Running denoising steps
4. Decoding to routing assignments
5. Evaluating with the critic

Run with: python scripts/demo_diffusion.py
"""

import sys
import os
import torch
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingDiffusion, RoutingState, create_routing_diffusion
from src.diffusion.sampler import initialize_routing_state, sample_routing
from src.critic.gnn import RoutingCritic, RoutingGraph
from src.critic.features import RoutingGraphBuilder
from src.critic.training import generate_synthetic_training_data
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.mcts.node import RoutingNode
from src.mcts.tree import MCTSTree


def create_toy_netlist(num_nets: int = 5, grid_size: int = 10) -> tuple:
    """Create a toy netlist for demo purposes."""
    nets = []
    for net_id in range(num_nets):
        num_pins = random.randint(2, 4)
        pins = []
        for pin_idx in range(num_pins):
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            pins.append(Pin(x=x, y=y, pin_id=pin_idx))
        nets.append(Net(net_id=net_id, pins=pins, name=f"net_{net_id}"))

    netlist = Netlist(nets=nets)
    grid = Grid(width=grid_size, height=grid_size)

    return netlist, grid


def compute_net_features(netlist: Netlist, grid: Grid, device: str = "cpu"):
    """Compute net features and positions for the diffusion model."""
    num_nets = len(netlist.nets)
    width, height = grid.get_size()

    features = torch.zeros(num_nets, 8, device=device)
    positions = torch.zeros(num_nets, 4, device=device)

    for i, net in enumerate(netlist.nets):
        if net.pins:
            xs = [p.x for p in net.pins]
            ys = [p.y for p in net.pins]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            features[i, 0] = len(net.pins) / 100.0  # fanout
            features[i, 1] = ((max_x - min_x) + (max_y - min_y)) / (width + height)  # HPWL

            positions[i, 0] = min_x / max(width, 1)
            positions[i, 1] = min_y / max(height, 1)
            positions[i, 2] = max_x / max(width, 1)
            positions[i, 3] = max_y / max(height, 1)

    return features, positions


def demo_diffusion_sampling():
    """Demo: Run diffusion model to generate routing candidates."""
    print("=" * 60)
    print("DEMO: Diffusion-based Routing Generation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create toy problem
    print("\n1. Creating toy netlist...")
    netlist, grid = create_toy_netlist(num_nets=5, grid_size=10)
    print(f"   Grid: {grid.width}x{grid.height}")
    print(f"   Nets: {len(netlist.nets)}")
    for net in netlist.nets:
        pins_str = ", ".join([f"({p.x},{p.y})" for p in net.pins])
        print(f"   - {net.name}: {pins_str}")

    # Create diffusion model
    print("\n2. Initializing diffusion model...")
    config = {
        "num_timesteps": 100,  # Reduced for demo
        "hidden_dim": 64,      # Smaller for demo
        "max_pips_per_net": 50,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1
    }
    model = create_routing_diffusion(config).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Num timesteps: {model.num_timesteps}")

    # Compute net features
    print("\n3. Computing net features...")
    net_features, net_positions = compute_net_features(netlist, grid, device)

    # Initialize state with noise
    print("\n4. Initializing routing state with noise...")
    pips_per_net = [20] * len(netlist.nets)  # Fixed for demo
    state = initialize_routing_state(
        num_nets=len(netlist.nets),
        pips_per_net=pips_per_net,
        num_timesteps=model.num_timesteps,
        device=device
    )
    print(f"   Initial timestep: {state.timestep}")
    print(f"   Routed nets: {len(state.routed_nets)}")
    print(f"   Net latents shape: {len(state.net_latents)} nets")

    # Run denoising steps
    print("\n5. Running denoising steps...")
    steps_to_run = min(20, model.num_timesteps)  # Limit for demo
    for step in range(steps_to_run):
        state = model.denoise_step(state, net_features, net_positions)
        if (step + 1) % 5 == 0:
            entropy = compute_average_entropy(state)
            print(f"   Step {step + 1}/{steps_to_run}: t={state.timestep}, "
                  f"routed={len(state.routed_nets)}, avg_entropy={entropy:.3f}")

    # Decode to routing assignment
    print("\n6. Decoding final routing...")
    routing = model.decode_routing(state, {})
    print(f"   Generated routing for {len(routing)} nets")
    for net_id, pips in routing.items():
        print(f"   - Net {net_id}: {len(pips)} PIPs selected")

    print("\n" + "=" * 60)
    print("Diffusion sampling complete!")
    print("=" * 60)

    return model, state, netlist, grid


def compute_average_entropy(state: RoutingState) -> float:
    """Compute average entropy across all net latents."""
    import torch.nn.functional as F

    total_entropy = 0.0
    count = 0
    for net_id, latent in state.net_latents.items():
        probs = F.softmax(latent, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        total_entropy += entropy
        count += 1

    return total_entropy / max(count, 1)


def demo_critic_evaluation():
    """Demo: Evaluate routing states with the critic."""
    print("\n" + "=" * 60)
    print("DEMO: Critic Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create synthetic training data
    print("\n1. Generating synthetic training examples...")
    examples = generate_synthetic_training_data(
        num_examples=10,
        grid_sizes=[(10, 10)],
        nets_range=(3, 8),
        device=device
    )
    print(f"   Generated {len(examples)} examples")

    # Create critic model
    print("\n2. Initializing critic model...")
    critic = RoutingCritic(
        node_dim=64,
        edge_dim=32,
        hidden_dim=64,
        num_layers=2
    ).to(device)
    critic.eval()

    num_params = sum(p.numel() for p in critic.parameters())
    print(f"   Critic parameters: {num_params:,}")

    # Evaluate examples
    print("\n3. Evaluating examples with critic...")
    graph_builder = RoutingGraphBuilder(Grid(10, 10))

    for i, example in enumerate(examples[:5]):
        graph = graph_builder.build_graph(example.state, example.netlist)
        graph = RoutingGraph(
            node_features=graph.node_features.to(device),
            edge_index=graph.edge_index.to(device),
            edge_features=graph.edge_features.to(device),
            congestion=graph.congestion.to(device),
            unrouted_mask=graph.unrouted_mask.to(device)
        )

        with torch.no_grad():
            predicted_score = critic(graph).item()

        print(f"   Example {i+1}: "
              f"timestep={example.state.timestep}, "
              f"routed={len(example.state.routed_nets)}/{len(example.netlist.nets)}, "
              f"target={example.final_score:.3f}, "
              f"predicted={predicted_score:.3f}")

    print("\n" + "=" * 60)
    print("Critic evaluation complete!")
    print("=" * 60)


def demo_mcts_tree():
    """Demo: Build an MCTS tree with routing nodes."""
    print("\n" + "=" * 60)
    print("DEMO: MCTS Tree Structure")
    print("=" * 60)

    device = "cpu"

    # Create toy problem
    netlist, grid = create_toy_netlist(num_nets=3, grid_size=8)

    # Initialize root state
    pips_per_net = [15] * len(netlist.nets)
    root_state = initialize_routing_state(
        num_nets=len(netlist.nets),
        pips_per_net=pips_per_net,
        num_timesteps=50,
        device=device
    )

    # Create root node
    print("\n1. Creating MCTS tree...")
    root = RoutingNode(state=root_state)
    tree = MCTSTree(root)

    # Simulate some tree expansion
    print("\n2. Simulating tree expansion...")
    model = create_routing_diffusion({
        "num_timesteps": 50,
        "hidden_dim": 32,
        "max_pips_per_net": 15,
        "num_heads": 2,
        "num_layers": 2
    })
    model.eval()

    net_features, net_positions = compute_net_features(netlist, grid, device)

    # Create a few children by denoising
    current = root
    for i in range(5):
        new_state = model.denoise_step(current.state, net_features, net_positions)
        child = RoutingNode(state=new_state, parent=current)
        current.children.append(child)

        # Simulate backprop
        reward = random.random()
        child.visit_count += 1
        child.total_value += reward

        current = child

    # Print tree stats
    stats = tree.get_statistics()
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Terminal nodes: {stats['terminal_nodes']}")
    print(f"   Pruned nodes: {stats['pruned_nodes']}")
    print(f"   Root visits: {stats['root_visits']}")
    print(f"   Root Q-value: {stats['root_Q']:.3f}")

    print("\n" + "=" * 60)
    print("MCTS tree demo complete!")
    print("=" * 60)


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  Diffusion-MCTS Routing Demo".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Run demos
    demo_diffusion_sampling()
    demo_critic_evaluation()
    demo_mcts_tree()

    print("\n" + "#" * 60)
    print("#" + " All demos completed successfully!".center(58) + "#")
    print("#" * 60)
    print("\nNext steps:")
    print("  1. Generate training data with real router (scripts/generate_routing_data.py)")
    print("  2. Train the diffusion model (train_diffusion.py)")
    print("  3. Train the critic (train_critic.py)")
    print("  4. Run MCTS search with trained models")


if __name__ == "__main__":
    main()
