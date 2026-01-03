"""DRC and timing constraints."""

from dataclasses import dataclass
from typing import List, Tuple, Set
from .state import RoutingState
from .grid import Grid


@dataclass
class DRCViolation:
    """Represents a DRC violation."""
    violation_type: str
    location: Tuple[int, int]
    net_id: int
    severity: float = 1.0


@dataclass
class Constraints:
    """Routing constraints."""
    min_spacing: int = 1
    min_width: int = 1
    max_congestion: float = 1.0
    blockages: Set[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Initialize blockages if not provided."""
        if self.blockages is None:
            self.blockages = set()
    
    def is_blocked(self, x: int, y: int) -> bool:
        """Check if a cell is blocked."""
        return (x, y) in self.blockages
    
    def check_spacing(self, state: RoutingState, x: int, y: int) -> bool:
        """Check spacing constraint."""
        # Check if cell is too close to other wires
        for wx, wy in state.occupied_cells:
            if (wx, wy) == (x, y):
                continue
            dx = abs(wx - x)
            dy = abs(wy - y)
            if dx + dy < self.min_spacing:
                return False
        return True
    
    def check_drc(self, state: RoutingState) -> List[DRCViolation]:
        """Check all DRC violations."""
        violations = []
        
        # Check blockages
        for x, y in state.occupied_cells:
            if self.is_blocked(x, y):
                violations.append(DRCViolation(
                    violation_type="blockage",
                    location=(x, y),
                    net_id=-1
                ))
        
        # Check spacing
        for wire in state.wires:
            if not self.check_spacing(state, wire.start[0], wire.start[1]):
                violations.append(DRCViolation(
                    violation_type="spacing",
                    location=wire.start,
                    net_id=wire.net_id
                ))
            if not self.check_spacing(state, wire.end[0], wire.end[1]):
                violations.append(DRCViolation(
                    violation_type="spacing",
                    location=wire.end,
                    net_id=wire.net_id
                ))
        
        return violations
    
    def get_congestion(self, state: RoutingState, grid: Grid) -> float:
        """Compute congestion metric."""
        if not state.occupied_cells:
            return 0.0
        total_cells = grid.width * grid.height
        occupied_ratio = len(state.occupied_cells) / total_cells
        return occupied_ratio

