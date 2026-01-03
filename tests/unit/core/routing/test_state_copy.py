"""Test routing state copying."""

import pytest
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.state import RoutingState, WireSegment


def test_deep_copy_independence():
    """Test that deep copy is independent."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    
    # Add wires
    state.add_wire(WireSegment(start=(0, 0), end=(1, 1), net_id=0))
    state.add_wire(WireSegment(start=(1, 1), end=(2, 2), net_id=0))
    
    # Copy
    state_copy = state.copy()
    
    # Verify independence
    assert state_copy.wires is not state.wires
    assert state_copy.occupied_cells is not state.occupied_cells
    assert state_copy.net_routes is not state.net_routes
    
    # Verify content equality
    assert len(state_copy.wires) == len(state.wires)
    assert state_copy.occupied_cells == state.occupied_cells
    assert state_copy.net_routes == state.net_routes

