"""Pytest fixtures for testing."""

import pytest
import torch
import numpy as np
from typing import Generator


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    return seed_value


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_grid():
    """Create a small 10x10 grid for testing."""
    from src.core.routing.grid import Grid
    return Grid(width=10, height=10, num_layers=1)


@pytest.fixture
def toy_netlist():
    """Create a toy netlist with 2 nets."""
    from src.core.routing.netlist import Netlist, Net, Pin
    nets = [
        Net(net_id=0, pins=[Pin(0, 0), Pin(9, 9)]),
        Net(net_id=1, pins=[Pin(0, 9), Pin(9, 0)]),
    ]
    return Netlist(nets=nets)


@pytest.fixture
def decoder():
    """Create a potential decoder."""
    from src.core.decoding.potential_decoder import PotentialDecoder
    return PotentialDecoder(latent_channels=1, output_channels=1)

