#!/usr/bin/env python3
"""Quick visualization of a single sample pickle file.

Usage:
    python scripts/visualize_single_sample.py /path/to/sample.pkl
"""

import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_single_sample.py <sample.pkl>")
        sys.exit(1)

    sample_path = sys.argv[1]
    print(f"Loading {sample_path}...")

    with open(sample_path, 'rb') as f:
        sample = pickle.load(f)

    # Extract data
    state = sample['routing_state']
    net_features = sample['net_features']
    net_positions = sample['net_positions']
    grid_size = sample.get('grid_size', 20)
    num_nets = sample.get('num_nets', len(state.net_latents))

    if isinstance(net_features, torch.Tensor):
        net_features = net_features.numpy()
    if isinstance(net_positions, torch.Tensor):
        net_positions = net_positions.numpy()

    print(f"\n=== Sample Info ===")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of nets: {num_nets}")
    print(f"Routed nets: {len(state.routed_nets)}")
    print(f"Timestep: {state.timestep}")

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # 1. Net placement
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title(f'Net Placement ({num_nets} nets)')
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')

    # Draw grid
    for i in range(grid_size + 1):
        ax1.axhline(y=i/grid_size, color='lightgray', linewidth=0.5)
        ax1.axvline(x=i/grid_size, color='lightgray', linewidth=0.5)

    # Draw net bounding boxes
    colors = plt.cm.tab20(np.linspace(0, 1, len(net_positions)))
    for i, (pos, color) in enumerate(zip(net_positions, colors)):
        x1, y1, x2, y2 = pos
        width = max(x2 - x1, 0.02)
        height = max(y2 - y1, 0.02)
        rect = patches.Rectangle((x1, y1), width, height,
                                  linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax1.add_patch(rect)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax1.text(cx, cy, str(i), ha='center', va='center', fontsize=8, fontweight='bold')

    # 2. Overall latent distribution
    ax2 = fig.add_subplot(2, 3, 2)
    all_latents = []
    for latent in state.net_latents.values():
        if isinstance(latent, torch.Tensor):
            latent = latent.numpy()
        all_latents.extend(latent.tolist())

    ax2.hist(all_latents, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='Selection threshold')
    ax2.set_xlabel('Latent Value')
    ax2.set_ylabel('Count')
    ax2.set_title('All Latents Distribution')
    ax2.legend()

    # Stats
    all_latents = np.array(all_latents)
    selected = (all_latents > 0).sum()
    total = len(all_latents)
    ax2.text(0.95, 0.95, f'Selected: {selected}/{total} ({100*selected/total:.1f}%)\nMean: {all_latents.mean():.2f}\nStd: {all_latents.std():.2f}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. PIPs per net
    ax3 = fig.add_subplot(2, 3, 3)
    net_ids = sorted(state.net_latents.keys())
    pip_counts = [len(state.net_latents[nid]) for nid in net_ids]
    ax3.bar(range(len(pip_counts)), pip_counts, color='steelblue')
    ax3.set_xlabel('Net ID')
    ax3.set_ylabel('Number of PIPs')
    ax3.set_title('PIPs per Net')

    # 4-6. Individual net latent distributions (first 3 nets)
    for idx in range(min(3, len(net_ids))):
        ax = fig.add_subplot(2, 3, 4 + idx)
        net_id = net_ids[idx]
        latent = state.net_latents[net_id]
        if isinstance(latent, torch.Tensor):
            latent = latent.numpy()

        ax.hist(latent, bins=30, edgecolor='black', alpha=0.7, color=colors[idx])
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Net {net_id} ({len(latent)} PIPs)')

        selected = (latent > 0).sum()
        ax.text(0.95, 0.95, f'{selected}/{len(latent)} selected',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Sample: {Path(sample_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = Path(sample_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")

    # Also show
    plt.show()


if __name__ == "__main__":
    main()
