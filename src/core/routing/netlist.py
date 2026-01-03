"""Netlist representation."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class Pin:
    """Represents a pin (connection point)."""
    x: int
    y: int
    pin_id: int = 0
    
    def __repr__(self) -> str:
        return f"Pin({self.x}, {self.y}, id={self.pin_id})"


@dataclass(frozen=True)
class Net:
    """Represents a net (connection between pins)."""
    net_id: int
    pins: List[Pin]
    name: str = ""
    
    def __post_init__(self):
        """Validate net has at least 2 pins."""
        if len(self.pins) < 2:
            raise ValueError(f"Net {self.net_id} must have at least 2 pins")
    
    @property
    def source(self) -> Pin:
        """Get source pin (first pin)."""
        return self.pins[0]
    
    @property
    def sinks(self) -> List[Pin]:
        """Get sink pins (all pins except source)."""
        return self.pins[1:]
    
    def __repr__(self) -> str:
        return f"Net(id={self.net_id}, pins={len(self.pins)})"


@dataclass(frozen=True)
class Netlist:
    """Represents a netlist (collection of nets)."""
    nets: List[Net]
    
    def __post_init__(self):
        """Validate netlist."""
        if not self.nets:
            raise ValueError("Netlist must contain at least one net")
        # Check for duplicate net IDs
        net_ids = [net.net_id for net in self.nets]
        if len(net_ids) != len(set(net_ids)):
            raise ValueError("Duplicate net IDs found")
    
    def get_net(self, net_id: int) -> Optional[Net]:
        """Get net by ID."""
        for net in self.nets:
            if net.net_id == net_id:
                return net
        return None
    
    def __len__(self) -> int:
        return len(self.nets)
    
    def __repr__(self) -> str:
        return f"Netlist({len(self.nets)} nets)"

