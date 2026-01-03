"""Test routing state hashing (should NOT be hashable)."""

import pytest
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.state import RoutingState, WireSegment


def test_state_not_hashable():
    """Test that RoutingState is NOT hashable (as per design)."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    
    # RoutingState should NOT be hashable
    with pytest.raises(TypeError):
        hash(state)


def test_state_copy():
    """Test that state copy produces identical state."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    
    # Add a wire
    wire = WireSegment(start=(0, 0), end=(1, 1), net_id=0)
    state.add_wire(wire)
    
    # Copy state
    state_copy = state.copy()
    
    # Check they have same wires
    assert len(state_copy.wires) == len(state.wires)
    assert state_copy.wires[0].start == state.wires[0].start
    assert state_copy.wires[0].end == state.wires[0].end
    
    # Check they have same occupied cells
    assert state_copy.occupied_cells == state.occupied_cells
    
    # Modify copy - should not affect original
    state_copy.add_wire(WireSegment(start=(2, 2), end=(3, 3), net_id=0))
    assert len(state.wires) == 1
    assert len(state_copy.wires) == 2

