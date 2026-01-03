"""Generate training data for routing critic.

Samples partial routing states from diffusion trajectories,
completes routing with nextpnr, records (partial_state, final_score) pairs.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

from src.diffusion.model import RoutingState, RoutingDiffusion
from src.diffusion.sampler import initialize_routing_state, sample_routing
from src.bridge.router import NextPNRRouter
from src.critic.training import RoutingExample
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist


def sample_partial_states(
    model: RoutingDiffusion,
    netlist: Netlist,
    grid: Grid,
    num_samples: int,
    num_partial_states_per_sample: int = 5
) -> List[RoutingExample]:
    """Sample partial routing states from diffusion trajectories.
    
    Args:
        model: Routing diffusion model
        netlist: Netlist
        grid: Grid
        num_samples: Number of full trajectories to sample
        num_partial_states_per_sample: Number of partial states per trajectory
    
    Returns:
        List of RoutingExample (partial state, final score) pairs
    """
    router = NextPNRRouter()
    examples = []
    
    # Compute net features (simplified)
    net_features_list = []
    net_positions_list = []
    num_pips_per_net = {}
    
    for net in netlist.nets:
        # Simple features
        fanout = len(net.pins) - 1
        features = torch.tensor([fanout, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, len(net.pins)], dtype=torch.float32)
        positions = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        
        net_features_list.append(features)
        net_positions_list.append(positions)
        num_pips_per_net[net.net_id] = max(10, fanout * 5)
    
    net_features = torch.stack(net_features_list) if net_features_list else torch.zeros(0, 8)
    net_positions = torch.stack(net_positions_list) if net_positions_list else torch.zeros(0, 4)
    
    netlist_info = {
        'net_features': net_features,
        'net_positions': net_positions,
        'num_pips_per_net': num_pips_per_net
    }
    
    for _ in tqdm(range(num_samples), desc="Sampling trajectories"):
        try:
            # Sample full routing trajectory
            final_state = sample_routing(model, netlist_info, device="cuda" if torch.cuda.is_available() else "cpu")
            
            # Extract partial states at different timesteps
            # For now, create synthetic partial states
            # In practice, would extract from actual diffusion trajectory
            
            # Create partial state by randomly routing some nets
            partial_state = initialize_routing_state(
                len(netlist.nets),
                [num_pips_per_net.get(net.net_id, 100) for net in netlist.nets],
                model.num_timesteps,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Route and get score
            # This is simplified - would need actual routing assignment
            routing_assignment = {}  # Would contain actual PIP assignments
            result = router.route_from_assignment(routing_assignment, netlist, grid)
            
            # Create example
            example = RoutingExample(
                state=partial_state,
                netlist=netlist,
                grid=grid,
                final_score=result.as_reward(),
                failed_nets=0 if result.success else 1,
                congestion_max=result.congestion,
                timing_slack=result.slack
            )
            
            examples.append(example)
            
        except Exception as e:
            print(f"Error sampling trajectory: {e}")
            continue
    
    return examples


def generate_critic_data(
    model_path: str,
    netlist_dir: str,
    output_dir: str,
    num_samples: int,
    nextpnr_path: str = "nextpnr-xilinx"
):
    """Generate critic training data.
    
    Args:
        model_path: Path to trained routing diffusion model
        netlist_dir: Directory with netlist files
        output_dir: Output directory for critic data
        num_samples: Number of samples to generate
        nextpnr_path: Path to nextpnr binary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = checkpoint.get('config', {}).get('model', {})
    model = RoutingDiffusion(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load netlists
    netlist_files = list(Path(netlist_dir).glob("*.json"))
    
    if not netlist_files:
        print(f"Warning: No netlist files found in {netlist_dir}")
        return
    
    # Process each netlist
    all_examples = []
    
    for netlist_file in tqdm(netlist_files[:num_samples // 10], desc="Processing netlists"):
        try:
            # Load netlist and grid (simplified - would use NextPNRReader)
            # For now, create dummy netlist/grid
            from src.core.routing.netlist import Netlist, Net, Pin
            from src.core.routing.grid import Grid
            
            # This is a placeholder - would load actual netlist
            netlist = Netlist(nets=[])
            grid = Grid(width=100, height=100)
            
            # Sample partial states
            examples = sample_partial_states(
                model,
                netlist,
                grid,
                num_samples=10,
                num_partial_states_per_sample=5
            )
            
            all_examples.extend(examples)
            
        except Exception as e:
            print(f"Error processing {netlist_file}: {e}")
            continue
    
    # Save examples
    output_file = output_path / "critic_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_examples, f)
    
    print(f"Generated {len(all_examples)} critic training examples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate critic training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained diffusion model")
    parser.add_argument("--netlist_dir", type=str, required=True, help="Directory with netlist files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for critic data")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--nextpnr_path", type=str, default="nextpnr-xilinx", help="Path to nextpnr binary")
    
    args = parser.parse_args()
    
    generate_critic_data(
        args.model_path,
        args.netlist_dir,
        args.output_dir,
        args.num_samples,
        args.nextpnr_path
    )


if __name__ == "__main__":
    main()

