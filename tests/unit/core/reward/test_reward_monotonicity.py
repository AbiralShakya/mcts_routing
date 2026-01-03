"""Test reward monotonicity."""

import pytest
from src.core.reward.reward import RewardFunction
from src.core.reward.metrics import compute_drc_violations
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.state import RoutingState, WireSegment
from src.core.routing.constraints import Constraints


def test_adding_drc_violation_decreases_reward():
    """Test that adding a DRC violation decreases reward (monotonicity)."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    constraints = Constraints(blockages={(5, 5)})
    reward_fn = RewardFunction()
    
    # Add wire without violation
    state.add_wire(WireSegment(start=(0, 0), end=(1, 1), net_id=0))
    reward1 = reward_fn.compute(state, constraints)
    
    # Add wire through blockage (violation)
    state.add_wire(WireSegment(start=(4, 4), end=(6, 6), net_id=0))
    reward2 = reward_fn.compute(state, constraints)
    
    # Reward should decrease
    assert reward2 < reward1


def test_reward_bounded():
    """Test that reward is bounded."""
    grid = Grid(width=10, height=10)
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    netlist = Netlist(nets=[net])
    state = RoutingState(grid=grid, netlist=netlist)
    constraints = Constraints()
    reward_fn = RewardFunction()
    
    reward = reward_fn.compute(state, constraints)
    
    # Reward should be finite
    assert not (reward == float('inf') or reward == float('-inf'))
    assert not (reward != reward)  # Not NaN

