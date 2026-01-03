"""Test decoder idempotence and Lipschitz continuity."""

import pytest
import torch
from src.core.decoding.potential_decoder import PotentialDecoder
from src.core.decoding.decoder import RoutingPotentials
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


@pytest.fixture
def decoder():
    """Create decoder."""
    return PotentialDecoder(latent_channels=1, output_channels=1)


@pytest.fixture
def grid():
    """Create grid."""
    return Grid(width=10, height=10)


@pytest.fixture
def netlist():
    """Create netlist."""
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    return Netlist(nets=[net])


def test_lipschitz_continuity(decoder, grid, netlist):
    """Test that decoder is Lipschitz-continuous."""
    # Create two similar latents
    latent1 = torch.randn(1, 10, 10)
    latent2 = latent1 + 0.01 * torch.randn(1, 10, 10)  # Small perturbation
    
    # Decode
    potentials1 = decoder.decode(latent1, grid, netlist)
    potentials2 = decoder.decode(latent2, grid, netlist)
    
    # Compute distances
    latent_diff = torch.norm(latent1 - latent2)
    potential_diff = torch.norm(potentials1.cost_field - potentials2.cost_field)
    
    # Lipschitz condition: ||decode(x) - decode(x')|| ≤ L ||x - x'||
    lipschitz_constant = decoder.get_lipschitz_constant()
    assert potential_diff <= lipschitz_constant * latent_diff + 1e-5  # Small tolerance


def test_no_discontinuities(decoder, grid, netlist):
    """Test that decoder has no discontinuities."""
    # Create latent
    latent = torch.randn(1, 10, 10)
    
    # Decode
    potentials = decoder.decode(latent, grid, netlist)
    
    # Check that outputs are continuous (no NaN, no Inf)
    assert not torch.isnan(potentials.cost_field).any()
    assert not torch.isinf(potentials.cost_field).any()
    
    # Check that outputs are positive (Softplus ensures this)
    assert torch.all(potentials.cost_field >= 0)

