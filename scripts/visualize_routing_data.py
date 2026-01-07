#!/usr/bin/env python3
"""Visualize routing training data.

Creates visualizations of:
1. Net placements on grid
2. Latent distributions (selected vs rejected PIPs)
3. Training data statistics
4. Sample comparisons

Usage:
    python scripts/visualize_routing_data.py \
        --data_dir data/routing_states \
        --output_dir visualizations \
        --num_samples 10
"""

import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import RoutingState


def load_samples(data_dir: str, num_samples: int = 10) -> List[dict]:
    """Load samples from data directory."""
    data_path = Path(data_dir)
    pkl_files = sorted(data_path.glob("*.pkl"))[:num_samples]

    samples = []
    for f in pkl_files:
        with open(f, 'rb') as fp:
            samples.append(pickle.load(fp))

    return samples


def visualize_net_placement(sample: dict, ax: plt.Axes, title: str = "Net Placement"):
    """Visualize net bounding boxes on grid."""
    grid_size = sample.get('grid_size', 20)
    net_positions = sample['net_positions']  # [num_nets, 4]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(y=i/grid_size, color='lightgray', linewidth=0.5)
        ax.axvline(x=i/grid_size, color='lightgray', linewidth=0.5)

    # Draw net bounding boxes
    colors = plt.cm.tab20(np.linspace(0, 1, len(net_positions)))

    for i, (pos, color) in enumerate(zip(net_positions, colors)):
        if isinstance(pos, torch.Tensor):
            pos = pos.numpy()

        x1, y1, x2, y2 = pos
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax.add_patch(rect)

        # Label
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, str(i), ha='center', va='center', fontsize=8, fontweight='bold')


def visualize_latent_distribution(sample: dict, ax: plt.Axes, net_idx: int = 0):
    """Visualize latent distribution for a specific net."""
    state = sample['routing_state']
    net_ids = sorted(state.net_latents.keys())

    if net_idx >= len(net_ids):
        net_idx = 0

    net_id = net_ids[net_idx]
    latent = state.net_latents[net_id]

    if isinstance(latent, torch.Tensor):
        latent = latent.numpy()

    # Histogram of latent values
    ax.hist(latent, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='Selection threshold')
    ax.set_xlabel('Latent Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Net {net_id} Latent Distribution')
    ax.legend()

    # Add statistics
    selected = (latent > 0).sum()
    total = len(latent)
    ax.text(0.95, 0.95, f'Selected: {selected}/{total}\nMean: {latent.mean():.2f}\nStd: {latent.std():.2f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def visualize_latent_heatmap(sample: dict, ax: plt.Axes, net_idx: int = 0):
    """Visualize latent as heatmap (reshape to 2D for visualization)."""
    state = sample['routing_state']
    net_ids = sorted(state.net_latents.keys())

    if net_idx >= len(net_ids):
        net_idx = 0

    net_id = net_ids[net_idx]
    latent = state.net_latents[net_id]

    if isinstance(latent, torch.Tensor):
        latent = latent.numpy()

    # Reshape to approximate square for visualization
    n = len(latent)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side)
    padded[:n] = latent
    grid = padded.reshape(side, side)

    im = ax.imshow(grid, cmap='RdYlGn', vmin=-5, vmax=5)
    ax.set_title(f'Net {net_id} Latent Heatmap')
    ax.set_xlabel('PIP Index (col)')
    ax.set_ylabel('PIP Index (row)')
    plt.colorbar(im, ax=ax, label='Latent Value')


def visualize_all_nets_summary(sample: dict, ax: plt.Axes):
    """Summary visualization of all nets' latent distributions."""
    state = sample['routing_state']
    net_ids = sorted(state.net_latents.keys())

    # Collect statistics
    means = []
    stds = []
    selection_ratios = []
    sizes = []

    for net_id in net_ids:
        latent = state.net_latents[net_id]
        if isinstance(latent, torch.Tensor):
            latent = latent.numpy()

        means.append(latent.mean())
        stds.append(latent.std())
        selection_ratios.append((latent > 0).mean())
        sizes.append(len(latent))

    x = np.arange(len(net_ids))
    width = 0.35

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, selection_ratios, width, label='Selection Ratio', color='steelblue')
    bars2 = ax2.bar(x + width/2, sizes, width, label='Num PIPs', color='coral', alpha=0.7)

    ax.set_xlabel('Net ID')
    ax.set_ylabel('Selection Ratio', color='steelblue')
    ax2.set_ylabel('Num PIPs', color='coral')
    ax.set_title('All Nets Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(net_ids)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')


def visualize_dataset_statistics(samples: List[dict], output_dir: Path):
    """Visualize overall dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of number of nets per sample
    num_nets = [s.get('num_nets', len(s['routing_state'].net_latents)) for s in samples]
    axes[0, 0].hist(num_nets, bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Nets')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Nets per Sample Distribution')

    # 2. Distribution of PIPs per net
    pips_per_net = []
    for s in samples:
        for latent in s['routing_state'].net_latents.values():
            pips_per_net.append(len(latent))

    axes[0, 1].hist(pips_per_net, bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Number of PIPs')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('PIPs per Net Distribution')

    # 3. Latent value distribution across all samples
    all_latents = []
    for s in samples:
        for latent in s['routing_state'].net_latents.values():
            if isinstance(latent, torch.Tensor):
                latent = latent.numpy()
            all_latents.extend(latent.tolist())

    axes[1, 0].hist(all_latents, bins=100, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Selection threshold')
    axes[1, 0].set_xlabel('Latent Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Overall Latent Distribution')
    axes[1, 0].legend()

    # 4. Selection ratio distribution
    selection_ratios = []
    for s in samples:
        for latent in s['routing_state'].net_latents.values():
            if isinstance(latent, torch.Tensor):
                latent = latent.numpy()
            selection_ratios.append((latent > 0).mean())

    axes[1, 1].hist(selection_ratios, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Selection Ratio')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Selection Ratio Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=150)
    plt.close()
    print(f"Saved dataset_statistics.png")


def visualize_net_features(samples: List[dict], output_dir: Path):
    """Visualize net feature distributions."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    feature_names = [
        'Fanout', 'BBox Width', 'BBox Height', 'HPWL',
        'Center X', 'Center Y', 'Pin Count', '(unused)'
    ]

    all_features = []
    for s in samples:
        feat = s['net_features']
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        all_features.append(feat)

    all_features = np.concatenate(all_features, axis=0)  # [total_nets, 7]

    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        if i < all_features.shape[1]:
            ax.hist(all_features[:, i], bins=30, edgecolor='black')
            ax.set_xlabel(name)
            ax.set_ylabel('Count')
            ax.set_title(f'Feature {i}: {name}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'net_features.png', dpi=150)
    plt.close()
    print(f"Saved net_features.png")


def visualize_single_sample(sample: dict, output_dir: Path, sample_idx: int):
    """Create detailed visualization for a single sample."""
    fig = plt.figure(figsize=(16, 12))

    # Grid spec for layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Net placement (large)
    ax1 = fig.add_subplot(gs[0, :2])
    visualize_net_placement(sample, ax1, f'Sample {sample_idx}: Net Placement')

    # 2. All nets summary
    ax2 = fig.add_subplot(gs[0, 2])
    visualize_all_nets_summary(sample, ax2)

    # 3-5. Individual net latent distributions
    state = sample['routing_state']
    net_ids = sorted(state.net_latents.keys())

    for i, net_idx in enumerate(range(min(3, len(net_ids)))):
        ax = fig.add_subplot(gs[1, i])
        visualize_latent_distribution(sample, ax, net_idx)

    # 6-8. Individual net latent heatmaps
    for i, net_idx in enumerate(range(min(3, len(net_ids)))):
        ax = fig.add_subplot(gs[2, i])
        visualize_latent_heatmap(sample, ax, net_idx)

    plt.suptitle(f'Sample {sample_idx} Detailed View', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / f'sample_{sample_idx:04d}_detail.png', dpi=150)
    plt.close()
    print(f"Saved sample_{sample_idx:04d}_detail.png")


def visualize_latent_comparison(samples: List[dict], output_dir: Path):
    """Compare latent distributions across multiple samples for same-ish nets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Collect first net's latent from each sample
    first_net_latents = []
    for s in samples[:6]:
        state = s['routing_state']
        net_ids = sorted(state.net_latents.keys())
        if net_ids:
            latent = state.net_latents[net_ids[0]]
            if isinstance(latent, torch.Tensor):
                latent = latent.numpy()
            first_net_latents.append(latent)

    for i, (ax, latent) in enumerate(zip(axes.flat, first_net_latents)):
        ax.hist(latent, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Sample {i} Net 0')

        selected = (latent > 0).sum()
        total = len(latent)
        ax.text(0.95, 0.95, f'{selected}/{total} selected',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Latent Distribution Comparison Across Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_comparison.png', dpi=150)
    plt.close()
    print(f"Saved latent_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize routing training data")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to load")
    parser.add_argument("--detailed", type=int, default=5, help="Number of detailed visualizations")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_samples} samples from {args.data_dir}...")
    samples = load_samples(args.data_dir, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # 1. Dataset statistics
    print("\nGenerating dataset statistics...")
    visualize_dataset_statistics(samples, output_dir)

    # 2. Net feature distributions
    print("Generating net feature distributions...")
    visualize_net_features(samples, output_dir)

    # 3. Latent comparison
    print("Generating latent comparison...")
    visualize_latent_comparison(samples, output_dir)

    # 4. Detailed single sample visualizations
    print(f"Generating {args.detailed} detailed sample visualizations...")
    for i in range(min(args.detailed, len(samples))):
        visualize_single_sample(samples[i], output_dir, i)

    print(f"\nAll visualizations saved to {output_dir}/")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
