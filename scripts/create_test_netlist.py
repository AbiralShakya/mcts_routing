#!/usr/bin/env python3
"""Create a simple test netlist JSON file for inference testing."""

import json
import argparse
from pathlib import Path
import random


def create_test_netlist(
    output_path: str,
    num_nets: int = 10,
    grid_width: int = 20,
    grid_height: int = 20,
    seed: int = 42
):
    """Create a simple test netlist in nextpnr JSON format.
    
    Args:
        output_path: Path to save the JSON file
        num_nets: Number of nets to create
        grid_width: Grid width
        grid_height: Grid height
        seed: Random seed
    """
    random.seed(seed)
    
    # Create netlist structure
    netlist_data = {
        "width": grid_width,
        "height": grid_height,
        "cells": [],
        "nets": []
    }
    
    # Create some cells (simplified - just for structure)
    for i in range(num_nets * 2):  # 2 cells per net (source + sink)
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        netlist_data["cells"].append({
            "name": f"cell_{i}",
            "x": x,
            "y": y,
            "bel": f"X{x}Y{y}"
        })
    
    # Create nets
    for net_id in range(num_nets):
        # Each net has 2-4 pins
        num_pins = random.randint(2, 4)
        
        # Create pins for this net
        pins = []
        for pin_idx in range(num_pins):
            x = random.randint(0, grid_width - 1)
            y = random.randint(0, grid_height - 1)
            pins.append({
                "pin_id": net_id * 100 + pin_idx,
                "x": x,
                "y": y
            })
        
        netlist_data["nets"].append({
            "name": f"net_{net_id}",
            "id": net_id,
            "pins": pins
        })
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(netlist_data, f, indent=2)
    
    print(f"Created test netlist: {output_path}")
    print(f"  Grid size: {grid_width}x{grid_height}")
    print(f"  Number of nets: {num_nets}")
    print(f"  Number of cells: {len(netlist_data['cells'])}")
    print(f"  Total pins: {sum(len(net['pins']) for net in netlist_data['nets'])}")


def main():
    parser = argparse.ArgumentParser(description="Create a test netlist JSON file")
    parser.add_argument("--output", type=str, default="data/test_netlist.json", 
                       help="Output file path")
    parser.add_argument("--num_nets", type=int, default=10, 
                       help="Number of nets")
    parser.add_argument("--grid_width", type=int, default=20, 
                       help="Grid width")
    parser.add_argument("--grid_height", type=int, default=20, 
                       help="Grid height")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    create_test_netlist(
        output_path=args.output,
        num_nets=args.num_nets,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

