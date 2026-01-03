"""FPGA grid representation (2D only)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class RoutingResource:
    """Represents a routing resource (wire, switch, etc.)."""
    resource_type: str
    capacity: int
    location: Tuple[int, int]  # (x, y) coordinates


@dataclass
class Grid:
    """2D FPGA grid representation.
    
    Note: This is 2D only (width, height). No depth dimension.
    """
    width: int
    height: int
    num_layers: int = 1  # Always 1 for 2D routing
    
    def __post_init__(self):
        """Validate grid dimensions."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.num_layers != 1:
            raise ValueError("Only 2D routing supported (num_layers=1)")
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if (x, y) is a valid position in the grid."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def get_size(self) -> Tuple[int, int]:
        """Get grid size as (width, height)."""
        return (self.width, self.height)
    
    def __repr__(self) -> str:
        return f"Grid(width={self.width}, height={self.height}, num_layers={self.num_layers})"

