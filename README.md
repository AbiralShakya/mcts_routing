# Diffusion-Guided MCTS Routing System

A research-grade diffusion-guided Monte Carlo Tree Search (MCTS) routing system for FPGA routing, with full integration to nextpnr/Xilinx tools.

## Overview

This system combines diffusion models with MCTS to solve FPGA routing problems. The key innovation is using diffusion models to generate soft routing potentials, which are then converted to hard routes via classical solvers. MCTS searches over diffusion trajectories to find optimal routing solutions.

## Architecture

- **Diffusion Model**: Generates soft routing potentials (cost fields, edge weights)
- **Classical Solver**: Converts soft potentials to hard routes (shortest-path, min-cost flow)
- **MCTS**: Searches over diffusion trajectories with time-aware value normalization
- **Integration**: Full nextpnr/Xilinx tool integration

## Key Features

- Soft potential decoder (Lipschitz-continuous)
- Time-aware MCTS value normalization
- Semantic branching control
- DDIM comparison framework
- Solver stability mechanisms (randomized tie-breaking)
- Value bootstrapping for variance reduction

## Installation

```bash
pip install -e .
```

## Usage

See `docs/` for detailed documentation.

## Development

Run tests:
```bash
pytest tests/
```

## License

[Add license]

