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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingDiffusion, create_routing_diffusion, RoutingState
from src.diffusion.sampler import initialize_routing_state
from src.bridge.router import NextPNRRouter
from src.critic.training import RoutingExample, generate_training_data, generate_synthetic_training_data
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.integration.nextpnr.reader import NextPNRReader
import yaml


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

    features = torch.tensor([
        fanout / 10.0,
        bbox_width / max(grid.width, 1),
        bbox_height / max(grid.height, 1),
        hpwl / (grid.width + grid.height),
        center_x,
        center_y,
        len(pins) / 10.0
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


def router_fn_wrapper(router: NextPNRRouter):
    """Create router function for generate_training_data."""
    def route_fn(routing_assignment, netlist, grid):
        try:
            result = router.route_from_assignment(routing_assignment, netlist, grid)
            return result
        except Exception as e:
            # Return failed result
            class FailedResult:
                success = False
                def as_reward(self):
                    return 0.0
                congestion = 1.0
                slack = -999.0
            return FailedResult()
    return route_fn


def generate_critic_data_from_diffusion(
    model_path: str,
    netlist_dir: str,
    output_dir: str,
    num_samples: int,
    nextpnr_path: str = "nextpnr-xilinx",
    use_synthetic: bool = False
):
    """Generate critic training data from diffusion model.
    
    Args:
        model_path: Path to trained routing diffusion model checkpoint
        netlist_dir: Directory with netlist files (or None for synthetic)
        output_dir: Output directory for critic data
        num_samples: Number of samples to generate
        nextpnr_path: Path to nextpnr binary
        use_synthetic: If True, use synthetic data instead of real netlists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load diffusion model
    if model_path and Path(model_path).exists():
        print(f"Loading diffusion model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model with same config
        model = create_routing_diffusion(model_config)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
    else:
        print("Warning: No model path provided, using synthetic data generation")
        use_synthetic = True
    
    if use_synthetic:
        # Generate synthetic training data
        print("Generating synthetic critic training data...")
        examples = generate_synthetic_training_data(
            num_examples=num_samples,
            grid_sizes=[(20, 20), (50, 50), (100, 100)],
            nets_range=(5, 30),
            device=device
        )
    else:
        # Load netlists
        router = NextPNRRouter(nextpnr_path=nextpnr_path)
        reader = NextPNRReader()
        
        netlist_files = list(Path(netlist_dir).glob("*.json"))
        if not netlist_files:
            print(f"Warning: No netlist files found in {netlist_dir}, using synthetic data")
            examples = generate_synthetic_training_data(
                num_examples=num_samples,
                device=device
            )
        else:
            # Process netlists
            netlists = []
            grids = []
            
            print(f"Loading {len(netlist_files)} netlists...")
            for netlist_file in tqdm(netlist_files[:min(10, len(netlist_files))], desc="Loading netlists"):
                try:
                    grid, netlist, placement, routing_state = reader.read_all(str(netlist_file))
                    if netlist and len(netlist.nets) > 0:
                        netlists.append(netlist)
                        grids.append(grid)
                except Exception as e:
                    print(f"Error loading {netlist_file}: {e}")
                    continue
            
            if not netlists:
                print("No valid netlists loaded, using synthetic data")
                examples = generate_synthetic_training_data(
                    num_examples=num_samples,
                    device=device
                )
            else:
                # Generate training data using diffusion model
                print(f"Generating {num_samples} samples from {len(netlists)} netlists...")
                router_fn = router_fn_wrapper(router)
                examples = generate_training_data(
                    netlists=netlists,
                    grids=grids,
                    router_fn=router_fn,
                    diffusion_model=model,
                    samples_per_netlist=max(1, num_samples // len(netlists)),
                    device=device
                )
    
    # Save examples
    output_file = output_path / "critic_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(examples, f)
    
    print(f"\nGenerated {len(examples)} critic training examples")
    print(f"Saved to {output_file}")
    
    # Print statistics
    if examples:
        scores = [ex.final_score for ex in examples]
        print(f"\nStatistics:")
        print(f"  Mean score: {sum(scores) / len(scores):.3f}")
        print(f"  Min score: {min(scores):.3f}")
        print(f"  Max score: {max(scores):.3f}")
        print(f"  Successful routings: {sum(1 for s in scores if s > 0.5)}/{len(scores)}")


def main():
    parser = argparse.ArgumentParser(description="Generate critic training data")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained diffusion model checkpoint")
    parser.add_argument("--netlist_dir", type=str, default=None, help="Directory with netlist files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for critic data")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--nextpnr_path", type=str, default="nextpnr-xilinx", help="Path to nextpnr binary")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data instead of real netlists")
    
    args = parser.parse_args()
    
    generate_critic_data_from_diffusion(
        model_path=args.model_path,
        netlist_dir=args.netlist_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        nextpnr_path=args.nextpnr_path,
        use_synthetic=args.use_synthetic
    )


if __name__ == "__main__":
    main()
