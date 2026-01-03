"""Individual reward metrics (wirelength, vias, DRC, congestion)."""

from typing import List
from ..routing.state import RoutingState
from ..routing.constraints import Constraints


def compute_wirelength(state: RoutingState) -> float:
    """Compute total wirelength.
    
    Args:
        state: Routing state
    
    Returns:
        Total wirelength (sum of all wire segment lengths)
    """
    total_length = 0.0
    for wire in state.wires:
        dx = abs(wire.end[0] - wire.start[0])
        dy = abs(wire.end[1] - wire.start[1])
        length = dx + dy  # Manhattan distance
        total_length += length
    return total_length


def compute_via_count(state: RoutingState) -> int:
    """Compute number of vias.
    
    Args:
        state: Routing state
    
    Returns:
        Number of vias (always 0 for 2D routing)
    """
    # 2D routing has no vias
    return 0


def compute_drc_violations(state: RoutingState, constraints: Constraints) -> int:
    """Compute number of DRC violations.
    
    Args:
        state: Routing state
        constraints: Routing constraints
    
    Returns:
        Number of DRC violations
    """
    violations = constraints.check_drc(state)
    return len(violations)


def compute_congestion(state: RoutingState, constraints: Constraints) -> float:
    """Compute congestion metric.
    
    Args:
        state: Routing state
        constraints: Routing constraints
    
    Returns:
        Congestion value [0, 1]
    """
    return constraints.get_congestion(state, state.grid)

