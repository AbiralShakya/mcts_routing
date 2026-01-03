"""Placement representation."""

from dataclasses import dataclass
from typing import Dict, Tuple
from .netlist import Net, Pin


@dataclass(frozen=True)
class Placement:
    """Represents placement of cells/pins on the grid."""
    pin_placements: Dict[int, Tuple[int, int]]  # pin_id -> (x, y)
    cell_placements: Dict[str, Tuple[int, int]] = None  # cell_name -> (x, y)
    
    def get_pin_position(self, pin_id: int) -> Tuple[int, int]:
        """Get position of a pin."""
        return self.pin_placements.get(pin_id, (-1, -1))
    
    def get_cell_position(self, cell_name: str) -> Tuple[int, int]:
        """Get position of a cell."""
        if self.cell_placements is None:
            return (-1, -1)
        return self.cell_placements.get(cell_name, (-1, -1))
    
    def __repr__(self) -> str:
        return f"Placement({len(self.pin_placements)} pins)"

