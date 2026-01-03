"""Test DRC detection in decoded potentials (placeholder - DRC is checked after solver)."""

import pytest
import torch
from src.core.decoding.potential_decoder import PotentialDecoder
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


def test_potentials_shape(decoder, small_grid, toy_netlist):
    """Test that decoded potentials match grid size."""
    decoder = PotentialDecoder()
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    
    latent = torch.randn(1, 10, 10)
    potentials = decoder.decode(latent, grid, netlist)
    
    # Cost field should match grid size
    assert potentials.cost_field.shape == (10, 10)

