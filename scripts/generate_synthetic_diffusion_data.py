#!/usr/bin/env python3
"""Generate MEANINGFUL synthetic training data for routing diffusion model.

The key insight: net_latents should encode WHICH PIPs are selected vs rejected.
- High logits (e.g., +5) for selected PIPs
- Low logits (e.g., -5) for rejected PIPs
- This creates a distribution the model can learn to predict

The old generator set everything to constant values (all 5.0 or all 0.0),
which provides zero learning signal.

Usage:
    python scripts/generate_synthetic_diffusion_data.py \
        --output_dir data/routing_states \
        --num_samples 2000 \
        --grid_size 20
"""

import argparse
import pickle
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingState
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


def create_synthetic_netlist(
    num_nets: int,
    grid_size: int,
    pins_range: tuple = (2, 6)
) -> tuple:
    """Create a synthetic netlist with random net placements."""
    nets = []
    for net_id in range(num_nets):
        num_pins = random.randint(pins_range[0], pins_range[1])
        pins = []
        for pin_idx in range(num_pins):
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            pins.append(Pin(x=x, y=y, pin_id=pin_idx))
        nets.append(Net(net_id=net_id, pins=pins, name=f"net_{net_id}"))

    netlist = Netlist(nets=nets)
    grid = Grid(width=grid_size, height=grid_size)
    return netlist, grid


def estimate_routing_pips(net: Net, grid: Grid) -> int:
    """Estimate number of PIPs needed to route a net based on HPWL."""
    if len(net.pins) < 2:
        return 10

    xs = [p.x for p in net.pins]
    ys = [p.y for p in net.pins]
    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))

    # Each unit of wirelength needs ~2 PIPs (forward + bend)
    # Plus some for Steiner tree branching
    fanout = len(net.pins) - 1
    estimated_pips = int(hpwl * 2) + fanout * 3

    return max(10, min(estimated_pips, 200))  # Clamp to reasonable range


def generate_routing_latent(
    num_pips: int,
    routing_difficulty: float = 0.5,
    selection_ratio: float = 0.2
) -> torch.Tensor:
    """Generate a meaningful routing latent distribution.

    Args:
        num_pips: Total number of candidate PIPs
        routing_difficulty: 0=easy (clear choices), 1=hard (ambiguous)
        selection_ratio: Fraction of PIPs to select

    Returns:
        Latent tensor where high values = selected PIPs
    """
    num_selected = max(3, int(num_pips * selection_ratio))

    # Start with base noise
    latent = torch.randn(num_pips) * 0.5

    # Select some PIPs to be the "chosen" ones
    selected_indices = random.sample(range(num_pips), num_selected)

    # Set logits based on difficulty
    # Easy: clear separation (selected=+5, rejected=-3)
    # Hard: less separation (selected=+2, rejected=-1)
    selected_logit = 5.0 - routing_difficulty * 3.0  # 5.0 to 2.0
    rejected_logit = -3.0 + routing_difficulty * 2.0  # -3.0 to -1.0

    for i in range(num_pips):
        if i in selected_indices:
            latent[i] = selected_logit + random.gauss(0, 0.3)
        else:
            latent[i] = rejected_logit + random.gauss(0, 0.3)

    return latent


def compute_net_features(net: Net, grid: Grid) -> torch.Tensor:
    """Compute 7-dim feature vector for a net."""
    pins = net.pins
    if len(pins) < 2:
        return torch.zeros(7)

    xs = [p.x for p in pins]
    ys = [p.y for p in pins]
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    fanout = len(pins) - 1
    hpwl = bbox_width + bbox_height
    center_x = (min(xs) + max(xs)) / 2.0 / max(grid.width, 1)
    center_y = (min(ys) + max(ys)) / 2.0 / max(grid.height, 1)

    # Normalized features
    features = torch.tensor([
        fanout / 10.0,                              # 0: normalized fanout
        bbox_width / max(grid.width, 1),            # 1: normalized bbox width
        bbox_height / max(grid.height, 1),          # 2: normalized bbox height
        hpwl / (grid.width + grid.height),          # 3: normalized HPWL
        center_x,                                   # 4: center x position
        center_y,                                   # 5: center y position
        len(pins) / 10.0                            # 6: normalized pin count
    ], dtype=torch.float32)

    return features


def compute_net_positions(net: Net, grid: Grid) -> torch.Tensor:
    """Compute normalized bounding box [x1, y1, x2, y2]."""
    pins = net.pins
    if len(pins) < 2:
        return torch.zeros(4)

    xs = [p.x for p in pins]
    ys = [p.y for p in pins]

    return torch.tensor([
        min(xs) / max(grid.width, 1),
        min(ys) / max(grid.height, 1),
        max(xs) / max(grid.width, 1),
        max(ys) / max(grid.height, 1)
    ], dtype=torch.float32)


def generate_sample(
    grid_size: int,
    nets_range: tuple = (5, 20),
    difficulty_range: tuple = (0.2, 0.8)
) -> dict:
    """Generate a single training sample with meaningful routing latents."""

    num_nets = random.randint(nets_range[0], nets_range[1])
    netlist, grid = create_synthetic_netlist(num_nets, grid_size)

    # Generate routing latents for each net
    net_latents = {}
    routed_nets = set()

    for net in netlist.nets:
        num_pips = estimate_routing_pips(net, grid)
        difficulty = random.uniform(difficulty_range[0], difficulty_range[1])

        # Generate meaningful latent distribution
        latent = generate_routing_latent(
            num_pips=num_pips,
            routing_difficulty=difficulty,
            selection_ratio=random.uniform(0.15, 0.3)
        )

        net_latents[net.net_id] = latent

        # Mark as routed with high probability (these are "solved" examples)
        if random.random() < 0.9:
            routed_nets.add(net.net_id)

    # Create routing state (timestep=0 means fully denoised/solved)
    routing_state = RoutingState(
        net_latents=net_latents,
        timestep=0,  # Terminal state
        routed_nets=routed_nets,
        congestion_map=None
    )

    # Compute net features and positions
    net_features = torch.stack([compute_net_features(net, grid) for net in netlist.nets])
    net_positions = torch.stack([compute_net_positions(net, grid) for net in netlist.nets])

    return {
        'routing_state': routing_state,
        'net_features': net_features,
        'net_positions': net_positions,
        'grid_size': grid_size,
        'num_nets': num_nets
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic diffusion training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size")
    parser.add_argument("--min_nets", type=int, default=5, help="Min nets per sample")
    parser.add_argument("--max_nets", type=int, default=20, help="Max nets per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} samples...")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Nets per sample: {args.min_nets}-{args.max_nets}")
    print(f"Output: {output_path}")

    for i in tqdm(range(args.num_samples)):
        sample = generate_sample(
            grid_size=args.grid_size,
            nets_range=(args.min_nets, args.max_nets)
        )

        output_file = output_path / f"sample_{i:06d}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(sample, f)

    print(f"\nGenerated {args.num_samples} samples in {output_path}")
    print("\nSample statistics:")

    # Load and show stats for first sample
    with open(output_path / "sample_000000.pkl", 'rb') as f:
        sample = pickle.load(f)

    state = sample['routing_state']
    print(f"  Nets: {len(state.net_latents)}")
    print(f"  Routed: {len(state.routed_nets)}")

    # Show latent distribution for first net
    first_net_id = list(state.net_latents.keys())[0]
    latent = state.net_latents[first_net_id]
    print(f"\n  First net latent stats:")
    print(f"    Shape: {latent.shape}")
    print(f"    Min: {latent.min().item():.2f}")
    print(f"    Max: {latent.max().item():.2f}")
    print(f"    Mean: {latent.mean().item():.2f}")
    print(f"    Std: {latent.std().item():.2f}")

    # Show how many PIPs would be selected (high logits)
    selected = (latent > 0).sum().item()
    print(f"    Selected PIPs (logit > 0): {selected}/{len(latent)}")

    print("\nData generation complete!")
    print("\nNext steps:")
    print("  1. Copy data to cluster")
    print("  2. Re-run training: sbatch scripts/slurm/train_diffusion.sh")
    print("  3. Loss should now decrease from ~1.0 toward lower values")


if __name__ == "__main__":
    main()
