"""Test grid representation."""

import pytest
from src.core.routing.grid import Grid


def test_grid_creation():
    """Test grid creation."""
    grid = Grid(width=10, height=10)
    assert grid.width == 10
    assert grid.height == 10
    assert grid.num_layers == 1


def test_grid_2d_only():
    """Test that only 2D grids are allowed."""
    with pytest.raises(ValueError):
        Grid(width=10, height=10, num_layers=2)


def test_is_valid_position():
    """Test position validation."""
    grid = Grid(width=10, height=10)
    assert grid.is_valid_position(0, 0)
    assert grid.is_valid_position(9, 9)
    assert not grid.is_valid_position(10, 10)
    assert not grid.is_valid_position(-1, -1)


def test_get_neighbors():
    """Test neighbor computation."""
    grid = Grid(width=10, height=10)
    neighbors = grid.get_neighbors(5, 5)
    assert len(neighbors) == 4
    assert (4, 5) in neighbors
    assert (6, 5) in neighbors
    assert (5, 4) in neighbors
    assert (5, 6) in neighbors
    
    # Corner should have 2 neighbors
    neighbors = grid.get_neighbors(0, 0)
    assert len(neighbors) == 2

