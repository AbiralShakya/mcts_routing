#!/usr/bin/env python3
"""Generate training data using ALGORITHMIC ROUTING.

The key insight: for diffusion to learn, there must be a learnable relationship
between net features/positions and which PIPs are selected.

Random PIP selection (previous approach) provides NO signal because:
- Net at position (0.1, 0.2) might randomly select PIPs [3, 7, 12]
- Net at position (0.1, 0.2) in another sample might select PIPs [1, 5, 19]
- The model has no way to learn which PIPs are "correct"

THIS generator uses simple A*/greedy routing to create actual valid routes:
- Net at position (0.1, 0.2) will consistently use PIPs near (0.1, 0.2)
- The model can learn: "nets in this region use these PIPs"

Usage:
    python scripts/generate_algorithmic_routing_data.py \
        --output_dir data/routing_states \
        --num_samples 5000 \
        --grid_size 20
"""

import argparse
import pickle
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
import heapq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingState
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


@dataclass
class PIP:
    """Programmable Interconnect Point."""
    pip_id: int
    x: int
    y: int
    direction: str  # 'H' or 'V' (horizontal/vertical)

    def __hash__(self):
        return hash((self.x, self.y, self.direction))


def create_pip_grid(grid_size: int) -> List[PIP]:
    """Create a grid of PIPs for routing.

    Each grid cell has:
    - 2 horizontal PIPs (in/out)
    - 2 vertical PIPs (in/out)
    """
    pips = []
    pip_id = 0

    for y in range(grid_size):
        for x in range(grid_size):
            # Horizontal PIPs
            pips.append(PIP(pip_id, x, y, 'H'))
            pip_id += 1
            # Vertical PIPs
            pips.append(PIP(pip_id, x, y, 'V'))
            pip_id += 1

    return pips


def get_pip_neighbors(pip: PIP, pips_by_pos: Dict, grid_size: int) -> List[PIP]:
    """Get neighboring PIPs that can be reached from this PIP."""
    neighbors = []
    x, y = pip.x, pip.y

    # Can switch direction at same position
    other_dir = 'V' if pip.direction == 'H' else 'H'
    if (x, y, other_dir) in pips_by_pos:
        neighbors.append(pips_by_pos[(x, y, other_dir)])

    # Can move in the current direction
    if pip.direction == 'H':
        # Move left or right
        if x > 0 and (x-1, y, 'H') in pips_by_pos:
            neighbors.append(pips_by_pos[(x-1, y, 'H')])
        if x < grid_size - 1 and (x+1, y, 'H') in pips_by_pos:
            neighbors.append(pips_by_pos[(x+1, y, 'H')])
    else:  # Vertical
        # Move up or down
        if y > 0 and (x, y-1, 'V') in pips_by_pos:
            neighbors.append(pips_by_pos[(x, y-1, 'V')])
        if y < grid_size - 1 and (x, y+1, 'V') in pips_by_pos:
            neighbors.append(pips_by_pos[(x, y+1, 'V')])

    return neighbors


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x2 - x1) + abs(y2 - y1)


def a_star_route(
    start: Tuple[int, int],
    end: Tuple[int, int],
    pips: List[PIP],
    pips_by_pos: Dict,
    grid_size: int,
    congestion: Optional[Dict] = None
) -> List[PIP]:
    """A* routing from start to end position.

    Args:
        start: (x, y) starting position
        end: (x, y) ending position
        pips: All available PIPs
        pips_by_pos: Dict mapping (x, y, dir) -> PIP
        grid_size: Grid dimension
        congestion: Optional congestion map for cost adjustment

    Returns:
        List of PIPs forming the route, or empty list if no route found
    """
    if congestion is None:
        congestion = {}

    # Start with both horizontal and vertical PIPs at start position
    start_pips = []
    for d in ['H', 'V']:
        if (start[0], start[1], d) in pips_by_pos:
            start_pips.append(pips_by_pos[(start[0], start[1], d)])

    if not start_pips:
        return []

    # Priority queue: (f_score, counter, g_score, pip_id, path)
    # Counter ensures unique ordering when f_scores are equal
    # f_score = g_score + heuristic
    open_set = []
    counter = 0
    for sp in start_pips:
        h = manhattan_distance(sp.x, sp.y, end[0], end[1])
        heapq.heappush(open_set, (h, counter, 0, sp.pip_id, [sp]))
        counter += 1

    closed_set = set()
    # Map pip_id -> PIP for reconstruction
    pip_by_id = {p.pip_id: p for p in pips}

    while open_set:
        f, _, g, current_id, path = heapq.heappop(open_set)
        current = pip_by_id.get(current_id) or path[-1]

        # Check if reached destination
        if current.x == end[0] and current.y == end[1]:
            return path

        if current_id in closed_set:
            continue
        closed_set.add(current_id)

        # Expand neighbors
        for neighbor in get_pip_neighbors(current, pips_by_pos, grid_size):
            if neighbor.pip_id in closed_set:
                continue

            # Cost: 1 base + congestion penalty
            cong_penalty = congestion.get((neighbor.x, neighbor.y), 0) * 2
            new_g = g + 1 + cong_penalty
            h = manhattan_distance(neighbor.x, neighbor.y, end[0], end[1])
            new_f = new_g + h

            heapq.heappush(open_set, (new_f, counter, new_g, neighbor.pip_id, path + [neighbor]))
            counter += 1

    return []  # No route found


def route_net(
    net: Net,
    pips: List[PIP],
    pips_by_pos: Dict,
    grid_size: int,
    congestion: Dict
) -> Tuple[List[PIP], bool]:
    """Route a multi-pin net using sequential A* routing.

    Uses simple approach: route from first pin to each subsequent pin.
    Updates congestion as routes are added.
    """
    if len(net.pins) < 2:
        return [], True

    all_route_pips = []

    # Route from first pin to each subsequent pin
    source = net.pins[0]
    for sink in net.pins[1:]:
        route = a_star_route(
            start=(source.x, source.y),
            end=(sink.x, sink.y),
            pips=pips,
            pips_by_pos=pips_by_pos,
            grid_size=grid_size,
            congestion=congestion
        )

        if not route:
            return all_route_pips, False

        all_route_pips.extend(route)

        # Update congestion
        for pip in route:
            key = (pip.x, pip.y)
            congestion[key] = congestion.get(key, 0) + 1

    return all_route_pips, True


def create_latent_from_route(
    route_pips: List[PIP],
    all_pips: List[PIP],
    selected_logit: float = 5.0,
    rejected_logit: float = -3.0,
    noise_std: float = 0.3
) -> torch.Tensor:
    """Convert a route (list of selected PIPs) to a latent representation.

    PIPs in the route get high logits, others get low logits.
    This creates a learnable distribution the diffusion model can predict.
    """
    latent = torch.zeros(len(all_pips))
    route_pip_ids = {p.pip_id for p in route_pips}

    for i, pip in enumerate(all_pips):
        if pip.pip_id in route_pip_ids:
            latent[i] = selected_logit + random.gauss(0, noise_std)
        else:
            latent[i] = rejected_logit + random.gauss(0, noise_std)

    return latent


def create_synthetic_netlist(
    num_nets: int,
    grid_size: int,
    pins_range: tuple = (2, 5)
) -> Netlist:
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

    return Netlist(nets=nets)


def compute_net_features(net: Net, grid_size: int) -> torch.Tensor:
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
    center_x = (min(xs) + max(xs)) / 2.0 / max(grid_size, 1)
    center_y = (min(ys) + max(ys)) / 2.0 / max(grid_size, 1)

    features = torch.tensor([
        fanout / 10.0,                          # 0: normalized fanout
        bbox_width / max(grid_size, 1),         # 1: normalized bbox width
        bbox_height / max(grid_size, 1),        # 2: normalized bbox height
        hpwl / (2 * grid_size),                 # 3: normalized HPWL
        center_x,                               # 4: center x position
        center_y,                               # 5: center y position
        len(pins) / 10.0                        # 6: normalized pin count
    ], dtype=torch.float32)

    return features


def compute_net_positions(net: Net, grid_size: int) -> torch.Tensor:
    """Compute normalized bounding box [x1, y1, x2, y2]."""
    pins = net.pins
    if len(pins) < 2:
        return torch.zeros(4)

    xs = [p.x for p in pins]
    ys = [p.y for p in pins]

    return torch.tensor([
        min(xs) / max(grid_size, 1),
        min(ys) / max(grid_size, 1),
        max(xs) / max(grid_size, 1),
        max(ys) / max(grid_size, 1)
    ], dtype=torch.float32)


def generate_sample(
    grid_size: int,
    nets_range: tuple = (5, 15),
    difficulty_range: tuple = (0.2, 0.8)
) -> dict:
    """Generate a single training sample with ALGORITHMIC routing."""

    num_nets = random.randint(nets_range[0], nets_range[1])
    netlist = create_synthetic_netlist(num_nets, grid_size)
    grid = Grid(width=grid_size, height=grid_size)

    # Create PIP infrastructure
    pips = create_pip_grid(grid_size)
    pips_by_pos = {(p.x, p.y, p.direction): p for p in pips}

    # Route all nets
    net_latents = {}
    routed_nets = set()
    congestion = {}

    # Vary routing difficulty
    difficulty = random.uniform(difficulty_range[0], difficulty_range[1])
    selected_logit = 5.0 - difficulty * 3.0  # 5.0 to 2.0
    rejected_logit = -3.0 + difficulty * 2.0  # -3.0 to -1.0

    for net in netlist.nets:
        # Get PIPs relevant to this net's bounding box (with margin)
        pins = net.pins
        xs = [p.x for p in pins]
        ys = [p.y for p in pins]
        min_x, max_x = max(0, min(xs) - 2), min(grid_size - 1, max(xs) + 2)
        min_y, max_y = max(0, min(ys) - 2), min(grid_size - 1, max(ys) + 2)

        # Filter PIPs to those in/near the net's region
        net_pips = [p for p in pips if min_x <= p.x <= max_x and min_y <= p.y <= max_y]

        if len(net_pips) < 10:
            net_pips = pips[:100]  # Fallback to first 100 PIPs

        # Route the net
        route_pips, success = route_net(
            net=net,
            pips=net_pips,
            pips_by_pos={(p.x, p.y, p.direction): p for p in net_pips},
            grid_size=grid_size,
            congestion=congestion
        )

        # Create latent from route
        # Re-index PIPs to be contiguous for this net
        pip_id_map = {p.pip_id: i for i, p in enumerate(net_pips)}
        remapped_route = []
        for p in route_pips:
            if p.pip_id in pip_id_map:
                remapped_route.append(PIP(pip_id_map[p.pip_id], p.x, p.y, p.direction))

        remapped_all = [PIP(i, p.x, p.y, p.direction) for i, p in enumerate(net_pips)]

        latent = create_latent_from_route(
            route_pips=remapped_route,
            all_pips=remapped_all,
            selected_logit=selected_logit,
            rejected_logit=rejected_logit,
            noise_std=0.3
        )

        net_latents[net.net_id] = latent

        if success:
            routed_nets.add(net.net_id)

    # Create routing state
    routing_state = RoutingState(
        net_latents=net_latents,
        timestep=0,  # Terminal state
        routed_nets=routed_nets,
        congestion_map=None
    )

    # Compute net features and positions
    net_features = torch.stack([compute_net_features(net, grid_size) for net in netlist.nets])
    net_positions = torch.stack([compute_net_positions(net, grid_size) for net in netlist.nets])

    return {
        'routing_state': routing_state,
        'net_features': net_features,
        'net_positions': net_positions,
        'grid_size': grid_size,
        'num_nets': num_nets,
        'num_routed': len(routed_nets)
    }


def main():
    parser = argparse.ArgumentParser(description="Generate algorithmic routing training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size")
    parser.add_argument("--min_nets", type=int, default=5, help="Min nets per sample")
    parser.add_argument("--max_nets", type=int, default=15, help="Max nets per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print("ALGORITHMIC ROUTING DATA GENERATOR")
    print(f"=" * 60)
    print(f"\nThis generator creates training data using A* routing.")
    print(f"Unlike random PIP selection, this creates LEARNABLE patterns.")
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Nets per sample: {args.min_nets}-{args.max_nets}")
    print(f"Output: {output_path}")

    total_nets = 0
    total_routed = 0

    for i in tqdm(range(args.num_samples)):
        sample = generate_sample(
            grid_size=args.grid_size,
            nets_range=(args.min_nets, args.max_nets)
        )

        total_nets += sample['num_nets']
        total_routed += sample['num_routed']

        output_file = output_path / f"sample_{i:06d}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(sample, f)

    print(f"\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"=" * 60)
    print(f"\nGenerated {args.num_samples} samples in {output_path}")
    print(f"\nRouting success rate: {total_routed}/{total_nets} ({100*total_routed/total_nets:.1f}%)")

    # Load and show stats for first sample
    with open(output_path / "sample_000000.pkl", 'rb') as f:
        sample = pickle.load(f)

    state = sample['routing_state']
    print(f"\nSample statistics (first sample):")
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

    print(f"\n" + "=" * 60)
    print("WHY THIS DATA IS BETTER")
    print(f"=" * 60)
    print("""
    Previous approach (random selection):
    - Net at (0.1, 0.2) selects PIPs [3, 7, 12] randomly
    - Same net position in next sample selects [1, 5, 19]
    - Model learns NOTHING because there's no pattern

    This approach (algorithmic routing):
    - Net at (0.1, 0.2) uses PIPs that connect its pins
    - Same net position will consistently use similar PIPs
    - Model learns: "nets in this region use these PIPs"

    Expected training behavior:
    - Loss should drop from ~1.0 to ~0.3-0.5 within 50 epochs
    - Validation loss should track training loss (not diverge)
    """)

    print("\nNext steps:")
    print("  1. Copy data to cluster: scp -r data/routing_states della:/path/")
    print("  2. Re-run training: sbatch scripts/slurm/train_joint.sh")
    print("  3. Watch loss - should decrease significantly!")


if __name__ == "__main__":
    main()
