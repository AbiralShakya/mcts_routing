"""Unit tests for bridge module."""

import pytest
import torch
import tempfile
import os

from src.bridge.router import NextPNRRouter, RoutingResult
from src.bridge.placement_io import (
    export_placement,
    import_routing,
    PlacementCache
)
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.placement import Placement


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_as_reward_success(self):
        """Test reward for successful routing."""
        result = RoutingResult(
            success=True,
            score=0.0,
            wirelength=1000,
            congestion=0.5,
            timing_met=True,
            slack=0.0,
            runtime_ms=100
        )

        reward = result.as_reward()
        assert 0 < reward <= 1.0

    def test_as_reward_failure(self):
        """Test reward for failed routing."""
        result = RoutingResult(
            success=False,
            score=0.0,
            wirelength=0,
            congestion=1.0,
            timing_met=False,
            slack=-10.0,
            runtime_ms=100,
            error_message="Routing failed"
        )

        assert result.as_reward() == 0.0

    def test_timing_impact(self):
        """Test that timing affects reward."""
        result_timing_met = RoutingResult(
            success=True, score=0.0, wirelength=1000,
            congestion=0.5, timing_met=True, slack=0.0, runtime_ms=100
        )
        result_timing_failed = RoutingResult(
            success=True, score=0.0, wirelength=1000,
            congestion=0.5, timing_met=False, slack=-5.0, runtime_ms=100
        )

        assert result_timing_met.as_reward() > result_timing_failed.as_reward()


class TestNextPNRRouter:
    """Tests for NextPNRRouter (subprocess mode only in tests)."""

    @pytest.fixture
    def router(self):
        return NextPNRRouter(
            nextpnr_path="nextpnr-xilinx",
            use_bindings=False,
            timeout_seconds=10
        )

    @pytest.fixture
    def grid(self):
        return Grid(width=50, height=50)

    @pytest.fixture
    def netlist(self):
        return Netlist(nets=[
            Net(net_id=0, pins=[
                Pin(x=10, y=10, pin_id=0),
                Pin(x=40, y=40, pin_id=1)
            ], name="test_net")
        ])

    @pytest.fixture
    def placement(self):
        return Placement(
            pin_placements={0: (10, 10), 1: (40, 40)},
            cell_placements={"cell0": (10, 10), "cell1": (40, 40)}
        )

    def test_export_placement_json(self, router, grid, netlist, placement):
        """Test JSON export for router."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "placement.json")
            router._export_placement_json(placement, netlist, grid, output_path)

            assert os.path.exists(output_path)

            # Check it's valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert "cells" in data
            assert "nets" in data


class TestPlacementIO:
    """Tests for placement I/O utilities."""

    @pytest.fixture
    def grid(self):
        return Grid(width=100, height=100)

    @pytest.fixture
    def netlist(self):
        return Netlist(nets=[
            Net(net_id=0, pins=[
                Pin(x=10, y=10, pin_id=0),
                Pin(x=50, y=50, pin_id=1)
            ], name="net0")
        ])

    @pytest.fixture
    def placement(self):
        return Placement(
            pin_placements={0: (10, 10), 1: (50, 50)},
            cell_placements={"c0": (10, 10), "c1": (50, 50)}
        )

    def test_export_json(self, grid, netlist, placement):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.json")
            export_placement(placement, netlist, grid, output_path, format="json")

            assert os.path.exists(output_path)

    def test_export_pcf(self, grid, netlist, placement):
        """Test PCF export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.pcf")
            export_placement(placement, netlist, grid, output_path, format="pcf")

            assert os.path.exists(output_path)

            with open(output_path) as f:
                content = f.read()

            assert "set_location" in content


class TestPlacementCache:
    """Tests for placement result cache."""

    @pytest.fixture
    def cache(self):
        return PlacementCache(max_size=10)

    @pytest.fixture
    def placement(self):
        return Placement(
            pin_placements={0: (1, 2), 1: (3, 4)},
            cell_placements={"a": (1, 2), "b": (3, 4)}
        )

    def test_cache_miss(self, cache, placement):
        """Test cache miss returns None."""
        assert cache.get(placement) is None

    def test_cache_hit(self, cache, placement):
        """Test cache hit returns stored value."""
        result = RoutingResult(
            success=True, score=0.8, wirelength=100,
            congestion=0.2, timing_met=True, slack=0.0,
            runtime_ms=50
        )

        cache.put(placement, result)
        cached = cache.get(placement)

        assert cached is not None
        assert cached.success == result.success
        assert cached.score == result.score

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = PlacementCache(max_size=3)

        # Fill cache
        for i in range(3):
            p = Placement(pin_placements={i: (i, i)}, cell_placements=None)
            r = RoutingResult(
                success=True, score=float(i), wirelength=i,
                congestion=0, timing_met=True, slack=0, runtime_ms=0
            )
            cache.put(p, r)

        # Add one more (should evict first)
        p_new = Placement(pin_placements={99: (99, 99)}, cell_placements=None)
        cache.put(p_new, RoutingResult(
            success=True, score=99, wirelength=99,
            congestion=0, timing_met=True, slack=0, runtime_ms=0
        ))

        # First should be evicted
        p_first = Placement(pin_placements={0: (0, 0)}, cell_placements=None)
        assert cache.get(p_first) is None

    def test_clear(self, cache, placement):
        """Test cache clear."""
        result = RoutingResult(
            success=True, score=0.5, wirelength=50,
            congestion=0.1, timing_met=True, slack=0, runtime_ms=10
        )
        cache.put(placement, result)
        cache.clear()

        assert cache.get(placement) is None
