#!/usr/bin/env python3
"""Simple inference script for trained diffusion model.

Usage:
    python scripts/run_inference.py \
        --checkpoint checkpoints/checkpoint_epoch_160.pt \
        --data_dir data/routing_states \
        --num_samples 5

This runs pure diffusion inference (no MCTS, no critic).
For full MCTS inference with critic pruning, train the critic first.
"""

import argparse
import torch
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingDiffusion, RoutingState, create_routing_diffusion
from src.diffusion.schedule import DDPMSchedule


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained diffusion model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Get model config
    model_config = config.get('model', {})
    model_config['net_feat_dim'] = model_config.get('net_feat_dim', 7)

    # Create model
    model = create_routing_diffusion(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return model, config


def load_sample(data_dir: str, idx: int = 0):
    """Load a sample from the dataset."""
    data_path = Path(data_dir)
    pkl_files = sorted(data_path.glob("*.pkl"))

    if not pkl_files:
        raise ValueError(f"No .pkl files found in {data_dir}")

    # Load one sample
    sample_file = pkl_files[idx % len(pkl_files)]
    with open(sample_file, 'rb') as f:
        sample = pickle.load(f)

    return sample, sample_file.name


def run_diffusion_inference(
    model: RoutingDiffusion,
    sample: dict,
    device: str = "cuda",
    num_steps: int = 50
):
    """Run diffusion inference on a sample.

    Args:
        model: Trained diffusion model
        sample: Sample dict with routing_state, net_features, net_positions
        device: Device to run on
        num_steps: Number of denoising steps (fewer = faster, more = better quality)

    Returns:
        Final routing state and decoded routing
    """
    # Extract sample data
    routing_state = sample['routing_state']
    net_features = sample['net_features'].to(device)
    net_positions = sample['net_positions'].to(device)

    # Handle batched vs unbatched
    if net_features.dim() == 2:
        net_features = net_features.unsqueeze(0)
        net_positions = net_positions.unsqueeze(0)

    # Create initial noisy state (start from pure noise)
    num_nets = len(routing_state.net_latents)
    net_ids = list(routing_state.net_latents.keys())

    # Initialize with noise
    noisy_latents = {}
    for net_id in net_ids:
        orig_len = len(routing_state.net_latents[net_id])
        noisy_latents[net_id] = torch.randn(orig_len, device=device)

    current_state = RoutingState(
        net_latents=noisy_latents,
        timestep=model.num_timesteps - 1,
        routed_nets=set(),
        congestion_map=None
    )

    # Denoise step by step
    step_size = max(1, model.num_timesteps // num_steps)

    print(f"Running {num_steps} denoising steps...")
    for i in range(num_steps):
        current_state = model.denoise_step(
            current_state,
            net_features,
            net_positions
        )

        # Update timestep (skip steps for faster inference)
        new_timestep = max(0, current_state.timestep - step_size + 1)
        current_state = RoutingState(
            net_latents=current_state.net_latents,
            timestep=new_timestep,
            routed_nets=current_state.routed_nets,
            congestion_map=current_state.congestion_map
        )

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{num_steps}, timestep={current_state.timestep}")

        if current_state.is_terminal():
            break

    # Decode to routing
    routing = model.decode_routing(current_state, {})

    return current_state, routing


def compare_with_ground_truth(
    predicted_routing: dict,
    ground_truth_state: RoutingState
):
    """Compare predicted routing with ground truth."""
    print("\n=== Comparison with Ground Truth ===")

    gt_nets = set(ground_truth_state.net_latents.keys())
    pred_nets = set(predicted_routing.keys())

    print(f"Ground truth nets: {len(gt_nets)}")
    print(f"Predicted nets: {len(pred_nets)}")
    print(f"Overlap: {len(gt_nets & pred_nets)}")

    # Compare PIP selections for common nets
    for net_id in sorted(gt_nets & pred_nets)[:3]:  # Show first 3
        gt_latent = ground_truth_state.net_latents[net_id]
        pred_pips = predicted_routing.get(net_id, [])

        # Get ground truth top PIPs
        gt_probs = torch.softmax(gt_latent, dim=-1)
        gt_top = gt_probs.topk(min(5, len(gt_probs))).indices.tolist()

        print(f"\nNet {net_id}:")
        print(f"  GT top PIPs: {gt_top}")
        print(f"  Predicted PIPs: {pred_pips[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Run diffusion inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to routing data directory")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to run inference on")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, args.device)

    # Run inference on samples
    for i in range(args.num_samples):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{args.num_samples}")
        print('='*50)

        # Load sample
        sample, filename = load_sample(args.data_dir, i)
        print(f"Loaded: {filename}")

        # Run inference
        final_state, routing = run_diffusion_inference(
            model, sample, args.device, args.num_steps
        )

        # Show results
        print(f"\nResults:")
        print(f"  Routed nets: {len(final_state.routed_nets)}/{len(final_state.net_latents)}")
        print(f"  Total PIPs selected: {sum(len(pips) for pips in routing.values())}")

        # Compare with ground truth
        compare_with_ground_truth(routing, sample['routing_state'])

    print("\n" + "="*50)
    print("Inference complete!")
    print("\nNext steps:")
    print("  1. Train critic model: sbatch scripts/slurm/train_critic.sh")
    print("  2. Run full MCTS inference with critic pruning")
    print("  3. Connect to nextpnr for real routing evaluation")


if __name__ == "__main__":
    main()
