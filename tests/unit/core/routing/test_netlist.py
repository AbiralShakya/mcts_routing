"""Test netlist representation."""

import pytest
from src.core.routing.netlist import Netlist, Net, Pin


def test_net_creation():
    """Test net creation."""
    pins = [Pin(0, 0), Pin(9, 9)]
    net = Net(net_id=0, pins=pins)
    assert net.net_id == 0
    assert len(net.pins) == 2


def test_net_requires_two_pins():
    """Test that net requires at least 2 pins."""
    with pytest.raises(ValueError):
        Net(net_id=0, pins=[Pin(0, 0)])


def test_netlist_creation():
    """Test netlist creation."""
    pins1 = [Pin(0, 0), Pin(9, 9)]
    pins2 = [Pin(0, 9), Pin(9, 0)]
    net1 = Net(net_id=0, pins=pins1)
    net2 = Net(net_id=1, pins=pins2)
    netlist = Netlist(nets=[net1, net2])
    assert len(netlist) == 2


def test_netlist_no_duplicate_ids():
    """Test that netlist rejects duplicate net IDs."""
    pins1 = [Pin(0, 0), Pin(9, 9)]
    pins2 = [Pin(0, 9), Pin(9, 0)]
    net1 = Net(net_id=0, pins=pins1)
    net2 = Net(net_id=0, pins=pins2)  # Duplicate ID
    with pytest.raises(ValueError):
        Netlist(nets=[net1, net2])


def test_get_net():
    """Test getting net by ID."""
    pins1 = [Pin(0, 0), Pin(9, 9)]
    pins2 = [Pin(0, 9), Pin(9, 0)]
    net1 = Net(net_id=0, pins=pins1)
    net2 = Net(net_id=1, pins=pins2)
    netlist = Netlist(nets=[net1, net2])
    
    assert netlist.get_net(0) == net1
    assert netlist.get_net(1) == net2
    assert netlist.get_net(2) is None

