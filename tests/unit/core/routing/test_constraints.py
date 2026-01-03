"""Test routing constraints."""

import pytest
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.state import RoutingState, WireSegment
from src.core.routing.constraints import Constraints, DRCViolation


def test_blockage_detection():
    """Test blockage detection."""
    grid = Grid(width=10, height=10)
    constraints = Constraints(blockages={(5, 5)})
    
    assert constraints.is_blocked(5, 5)
    assert not constraints.is_blocked(5, 6)


def test_drc_violations():
    """Test DRC violation detection."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    
    constraints = Constraints(blockages={(5, 5)}, min_spacing=2)
    
    # Add wire that starts at a blockage
    state.add_wire(WireSegment(start=(5, 5), end=(6, 6), net_id=0))
    
    violations = constraints.check_drc(state)
    assert len(violations) > 0
    assert any(v.violation_type == "blockage" for v in violations)


def test_congestion_metric():
    """Test congestion computation."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    constraints = Constraints()
    
    # Empty state should have zero congestion
    congestion = constraints.get_congestion(state, grid)
    assert congestion == 0.0
    
    # Add wires
    state.add_wire(WireSegment(start=(0, 0), end=(1, 1), net_id=0))
    congestion = constraints.get_congestion(state, grid)
    assert congestion > 0.0
    assert congestion <= 1.0

