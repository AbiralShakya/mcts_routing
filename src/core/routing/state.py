"""Routing state representation.

Note: RoutingState is NOT hashable in MCTS context.
It's a derived artifact used only for reward computation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np

from typing import TYPE_CHECKING

from .grid import Grid
from .netlist import Net, Pin

if TYPE_CHECKING:
    from .netlist import Netlist


@dataclass
class WireSegment:
    """Represents a wire segment in the routing."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    net_id: int
    layer: int = 0  # Always 0 for 2D routing


@dataclass
class RoutingState:
    """Routing state (mutable, NOT hashable).
    
    This is used for:
    - Reward computation
    - Solver updates
    - NOT for MCTS node identity (only latents are hashed)
    """
    grid: Grid
    netlist: "Netlist"  # Forward reference
    wires: List[WireSegment] = field(default_factory=list)
    occupied_cells: Set[Tuple[int, int]] = field(default_factory=set)
    net_routes: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    
    def add_wire(self, wire: WireSegment) -> None:
        """Add a wire segment to the routing."""
        self.wires.append(wire)
        # Update occupied cells
        self.occupied_cells.add(wire.start)
        self.occupied_cells.add(wire.end)
        # Update net routes
        if wire.net_id not in self.net_routes:
            self.net_routes[wire.net_id] = []
        if wire.start not in self.net_routes[wire.net_id]:
            self.net_routes[wire.net_id].append(wire.start)
        if wire.end not in self.net_routes[wire.net_id]:
            self.net_routes[wire.net_id].append(wire.end)
    
    def is_cell_occupied(self, x: int, y: int) -> bool:
        """Check if a cell is occupied."""
        return (x, y) in self.occupied_cells
    
    def get_net_route(self, net_id: int) -> List[Tuple[int, int]]:
        """Get routing path for a net."""
        return self.net_routes.get(net_id, [])
    
    def is_net_routed(self, net_id: int) -> bool:
        """Check if a net is fully routed."""
        if net_id not in self.net_routes:
            return False
        net = self.netlist.get_net(net_id)
        if net is None:
            return False
        # Check if all pins are connected
        route = self.net_routes[net_id]
        pin_positions = {(pin.x, pin.y) for pin in net.pins}
        route_set = set(route)
        return pin_positions.issubset(route_set)
    
    def copy(self) -> 'RoutingState':
        """Create a deep copy of the routing state."""
        return RoutingState(
            grid=self.grid,
            netlist=self.netlist,
            wires=self.wires.copy(),
            occupied_cells=self.occupied_cells.copy(),
            net_routes={k: v.copy() for k, v in self.net_routes.items()}
        )
    
    def __repr__(self) -> str:
        routed_nets = sum(1 for net_id in self.net_routes.keys() 
                         if self.is_net_routed(net_id))
        return f"RoutingState({routed_nets}/{len(self.netlist)} nets routed, {len(self.wires)} wires)"

