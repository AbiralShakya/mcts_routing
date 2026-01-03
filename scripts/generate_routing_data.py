"""Generate training data for routing diffusion model.

Loads nextpnr designs, runs router, converts to RoutingState format.
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from tqdm import tqdm

from src.integration.nextpnr.reader import NextPNRReader
from src.bridge.router import NextPNRRouter
from src.diffusion.model import RoutingState
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net


def compute_net_features(net: Net, grid: Grid) -> torch.Tensor:
    """Compute features for a net.
    
    Returns:
        Feature vector [8]: fanout, bbox_width, bbox_height, criticality, etc.
    """
    pins = net.pins
    if len(pins) < 2:
        return torch.zeros(8)
    
    # Bounding box
    x_coords = [pin.x for pin in pins]
    y_coords = [pin.y for pin in pins]
    bbox_width = max(x_coords) - min(x_coords)
    bbox_height = max(y_coords) - min(y_coords)
    
    # Fanout
    fanout = len(pins) - 1
    
    # Normalize by grid size
    bbox_width_norm = bbox_width / max(grid.width, 1)
    bbox_height_norm = bbox_height / max(grid.height, 1)
    
    # Manhattan distance
    manhattan = abs(max(x_coords) - min(x_coords)) + abs(max(y_coords) - min(y_coords))
    manhattan_norm = manhattan / (grid.width + grid.height)
    
    # Center position (normalized)
    center_x = (min(x_coords) + max(x_coords)) / 2.0 / max(grid.width, 1)
    center_y = (min(y_coords) + max(y_coords)) / 2.0 / max(grid.height, 1)
    
    # Criticality (placeholder - would come from timing analysis)
    criticality = 0.5
    
    features = torch.tensor([
        fanout,
        bbox_width_norm,
        bbox_height_norm,
        manhattan_norm,
        center_x,
        center_y,
        criticality,
        len(pins)  # Total pins
    ], dtype=torch.float32)
    
    return features


def compute_net_positions(net: Net, grid: Grid) -> torch.Tensor:
    """Compute normalized bounding box for net.
    
    Returns:
        [x1, y1, x2, y2] normalized to [0, 1]
    """
    pins = net.pins
    if len(pins) < 2:
        return torch.zeros(4)
    
    x_coords = [pin.x for pin in pins]
    y_coords = [pin.y for pin in pins]
    
    x1 = min(x_coords) / max(grid.width, 1)
    y1 = min(y_coords) / max(grid.height, 1)
    x2 = max(x_coords) / max(grid.width, 1)
    y2 = max(y_coords) / max(grid.height, 1)
    
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def routing_to_routing_state(
    routing_result: Dict,
    netlist: Netlist,
    grid: Grid
) -> RoutingState:
    """Convert nextpnr routing result to RoutingState.
    
    Args:
        routing_result: Routing assignment from nextpnr
        netlist: Netlist
        grid: Grid
    
    Returns:
        RoutingState with net latents encoding the routing
    """
    net_latents = {}
    routed_nets = set()
    
    # For each net, create latent distribution over PIPs
    # In practice, this would map actual PIPs to indices
    # For now, create a simple encoding
    for net in netlist.nets:
        net_id = net.net_id
        
        # Get routing for this net (if available)
        if net_id in routing_result:
            # Net is routed - create concentrated distribution
            num_pips = len(routing_result[net_id])
            # Create one-hot-like distribution (high confidence)
            latent = torch.zeros(num_pips)
            # Set high values for routed PIPs
            latent[:] = 5.0  # High logit = committed
            routed_nets.add(net_id)
        else:
            # Net not routed - create uniform distribution
            # Estimate number of feasible PIPs (simplified)
            num_pips = max(10, len(net.pins) * 5)
            latent = torch.zeros(num_pips)  # Uniform = low confidence
        
        net_latents[net_id] = latent
    
    return RoutingState(
        net_latents=net_latents,
        timestep=0,  # Terminal state (fully routed)
        routed_nets=routed_nets,
        congestion_map=None
    )


def generate_routing_data(
    input_dir: str,
    output_dir: str,
    num_samples: int,
    nextpnr_path: str = "nextpnr-xilinx"
):
    """Generate routing training data.
    
    Args:
        input_dir: Directory with nextpnr design files
        output_dir: Output directory for training data
        num_samples: Number of samples to generate
        nextpnr_path: Path to nextpnr binary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find design files
    design_files = list(input_path.glob("*.json")) + list(input_path.glob("*.fasm"))
    
    if not design_files:
        print(f"Warning: No design files found in {input_dir}")
        return
    
    reader = NextPNRReader()
    router = NextPNRRouter(nextpnr_path=nextpnr_path)
    
    samples_generated = 0
    file_idx = 0
    
    # Split files for train/val/test
    train_files = []
    val_files = []
    test_files = []
    
    for i, design_file in enumerate(design_files):
        if i < int(0.9 * len(design_files)):
            train_files.append(design_file)
        elif i < int(0.95 * len(design_files)):
            val_files.append(design_file)
        else:
            test_files.append(design_file)
    
    # Write split files
    with open(output_path / "train_files.txt", 'w') as f:
        f.write('\n'.join([f.name for f in train_files]))
    with open(output_path / "val_files.txt", 'w') as f:
        f.write('\n'.join([f.name for f in val_files]))
    with open(output_path / "test_files.txt", 'w') as f:
        f.write('\n'.join([f.name for f in test_files]))
    
    # Process designs
    all_files = train_files + val_files + test_files
    
    for design_file in tqdm(all_files[:num_samples], desc="Generating routing data"):
        try:
            # Read design
            grid, netlist, placement, routing_state = reader.read_all(str(design_file))
            
            # If no routing, try to route it
            if routing_state is None:
                # Route with nextpnr
                result = router.route(placement, netlist, grid)
                if not result.success:
                    continue  # Skip failed routings
                
                # Convert routing result to RoutingState
                # This is simplified - would need actual PIP mapping
                routing_state = routing_to_routing_state(
                    {},  # Would contain actual routing assignment
                    netlist,
                    grid
                )
            
            # Compute net features and positions
            net_features_list = []
            net_positions_list = []
            
            for net in netlist.nets:
                features = compute_net_features(net, grid)
                positions = compute_net_positions(net, grid)
                net_features_list.append(features)
                net_positions_list.append(positions)
            
            # Stack into tensors
            net_features = torch.stack(net_features_list) if net_features_list else torch.zeros(0, 8)
            net_positions = torch.stack(net_positions_list) if net_positions_list else torch.zeros(0, 4)
            
            # Create data sample
            sample = {
                'routing_state': routing_state,
                'net_features': net_features.numpy(),
                'net_positions': net_positions.numpy(),
                'grid': grid,
                'netlist': netlist
            }
            
            # Save sample
            output_file = output_path / f"sample_{file_idx:06d}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(sample, f)
            
            samples_generated += 1
            file_idx += 1
            
        except Exception as e:
            print(f"Error processing {design_file}: {e}")
            continue
    
    print(f"Generated {samples_generated} routing samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate routing training data")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with nextpnr designs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training data")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--nextpnr_path", type=str, default="nextpnr-xilinx", help="Path to nextpnr binary")
    
    args = parser.parse_args()
    
    generate_routing_data(
        args.input_dir,
        args.output_dir,
        args.num_samples,
        args.nextpnr_path
    )


if __name__ == "__main__":
    main()

