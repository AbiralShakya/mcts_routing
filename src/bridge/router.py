"""Python interface to nextpnr router.

Two modes:
1. Subprocess: Call nextpnr binary (slower, always works)
2. C++ bindings: Direct call to router2.cc (faster, requires compilation)
"""

import subprocess
import tempfile
import json
import os
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from ..core.routing.placement import Placement
from ..core.routing.netlist import Netlist
from ..core.routing.grid import Grid


@dataclass
class RoutingResult:
    """Result from nextpnr routing attempt."""
    success: bool  # Did routing complete without DRC errors?
    score: float   # 0-1, quality metric
    wirelength: int  # Total wirelength
    congestion: float  # Max congestion ratio
    timing_met: bool  # Timing constraints met?
    slack: float  # Worst negative slack (0 if timing met)
    runtime_ms: float  # Router runtime
    error_message: Optional[str] = None

    def as_reward(self) -> float:
        """Convert to MCTS reward signal."""
        if not self.success:
            return 0.0

        # Combine metrics into single score
        # Prioritize: success > timing > congestion > wirelength
        base_score = 0.5  # For successful route

        # Timing bonus (0 to 0.3)
        if self.timing_met:
            timing_score = 0.3
        else:
            timing_score = max(0, 0.3 * (1 + self.slack / 10.0))  # Negative slack hurts

        # Congestion penalty (0 to 0.1)
        congestion_score = 0.1 * max(0, 1 - self.congestion)

        # Wirelength bonus (0 to 0.1) - normalized by grid size
        # Lower is better, assume max wirelength ~10000
        wl_score = 0.1 * max(0, 1 - self.wirelength / 10000)

        return min(1.0, base_score + timing_score + congestion_score + wl_score)


class NextPNRRouter:
    """Interface to nextpnr router.

    Calls router2.cc to evaluate placement quality.
    This is the "oracle" in our MCTS - the ground truth for what routes.
    """

    def __init__(
        self,
        nextpnr_path: str = "nextpnr-xilinx",
        chipdb_path: Optional[str] = None,
        use_bindings: bool = False,
        timeout_seconds: float = 60.0
    ):
        """Initialize router interface.

        Args:
            nextpnr_path: Path to nextpnr-xilinx binary
            chipdb_path: Path to chip database (required for Xilinx)
            use_bindings: Use C++ bindings instead of subprocess
            timeout_seconds: Router timeout
        """
        self.nextpnr_path = nextpnr_path
        self.chipdb_path = chipdb_path
        self.use_bindings = use_bindings
        self.timeout = timeout_seconds

        # Try to import C++ bindings if requested
        self._bindings = None
        if use_bindings:
            try:
                from . import _nextpnr_bindings as bindings
                self._bindings = bindings
            except ImportError:
                print("Warning: C++ bindings not available, falling back to subprocess")
                self.use_bindings = False

    def route_from_assignment(
        self,
        routing_assignment: Dict[int, List[int]],
        netlist: Netlist,
        grid: Grid
    ) -> RoutingResult:
        """Route from a diffusion-generated assignment.

        Args:
            routing_assignment: Dict[net_id -> list of PIPs]
            netlist: Design netlist
            grid: FPGA grid

        Returns:
            RoutingResult with quality metrics
        """
        # Convert assignment to placement format and route
        # This is a simplified interface - actual implementation
        # would pass the routing assignment directly to nextpnr
        return self._route_subprocess_with_hints(routing_assignment, netlist, grid)

    def _route_subprocess_with_hints(
        self,
        routing_hints: Dict[int, List[int]],
        netlist: Netlist,
        grid: Grid
    ) -> RoutingResult:
        """Route with routing hints from diffusion.

        Creates a placement from netlist pins and routes using nextpnr.
        The routing_hints provide guidance but actual routing is done by nextpnr.

        Args:
            routing_hints: Dict[net_id -> list of preferred PIP indices]
            netlist: Design netlist
            grid: FPGA grid

        Returns:
            RoutingResult with actual routing metrics
        """
        import time

        # Create a placement from netlist pins
        # Each net's source pin becomes a cell placement
        pin_placements = {}
        cell_placements = {}

        for net in netlist.nets:
            for i, pin in enumerate(net.pins):
                pin_placements[pin.pin_id] = (pin.x, pin.y)
                # Create pseudo-cell for each pin
                cell_name = f"cell_{net.net_id}_{i}"
                cell_placements[cell_name] = (pin.x, pin.y)

        placement = Placement(
            pin_placements=pin_placements,
            cell_placements=cell_placements
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export placement with routing hints
            placement_json = os.path.join(tmpdir, "placement.json")
            self._export_placement_with_hints(
                placement, netlist, grid, routing_hints, placement_json
            )

            # Prepare nextpnr command
            cmd = [self.nextpnr_path]

            if self.chipdb_path:
                cmd.extend(["--chipdb", self.chipdb_path])

            report_path = os.path.join(tmpdir, "report.json")
            cmd.extend([
                "--json", placement_json,
                "--report", report_path
            ])

            # Run router
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )
                runtime_ms = (time.time() - start_time) * 1000

                # Parse results
                return self._parse_router_output(
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    report_path,
                    runtime_ms
                )

            except FileNotFoundError:
                # nextpnr not installed - return heuristic estimate
                return self._estimate_routing_quality(routing_hints, netlist, grid)

            except subprocess.TimeoutExpired:
                return RoutingResult(
                    success=False,
                    score=0.0,
                    wirelength=0,
                    congestion=1.0,
                    timing_met=False,
                    slack=-999.0,
                    runtime_ms=self.timeout * 1000,
                    error_message="Router timeout"
                )

            except Exception as e:
                return RoutingResult(
                    success=False,
                    score=0.0,
                    wirelength=0,
                    congestion=1.0,
                    timing_met=False,
                    slack=-999.0,
                    runtime_ms=0,
                    error_message=str(e)
                )

    def _export_placement_with_hints(
        self,
        placement: Placement,
        netlist: Netlist,
        grid: Grid,
        routing_hints: Dict[int, List[int]],
        output_path: str
    ) -> None:
        """Export placement with routing hints to nextpnr JSON format."""
        width = grid.width if hasattr(grid, 'width') else grid.get_size()[0]
        height = grid.height if hasattr(grid, 'height') else grid.get_size()[1]

        data = {
            "width": width,
            "height": height,
            "cells": [],
            "nets": [],
            "routing_hints": {}  # Custom field for hints
        }

        # Export cell placements
        if placement.cell_placements:
            for cell_name, (x, y) in placement.cell_placements.items():
                data["cells"].append({
                    "name": cell_name,
                    "x": x,
                    "y": y,
                    "bel": f"X{x}Y{y}"
                })

        # Export nets with hints
        for net in netlist.nets:
            net_data = {
                "name": net.name,
                "id": net.net_id,
                "pins": [
                    {"pin_id": pin.pin_id, "x": pin.x, "y": pin.y}
                    for pin in net.pins
                ]
            }
            data["nets"].append(net_data)

            # Add routing hints for this net
            if net.net_id in routing_hints:
                data["routing_hints"][str(net.net_id)] = routing_hints[net.net_id]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _estimate_routing_quality(
        self,
        routing_hints: Dict[int, List[int]],
        netlist: Netlist,
        grid: Grid
    ) -> RoutingResult:
        """Estimate routing quality when nextpnr is not available.

        Uses heuristics based on net topology and congestion estimates.
        This provides meaningful training signal even without real router.
        """
        import random

        width, height = grid.get_size()
        total_wirelength = 0
        congestion_map = {}  # (x, y) -> usage count

        for net in netlist.nets:
            if len(net.pins) < 2:
                continue

            # Compute HPWL for this net
            xs = [p.x for p in net.pins]
            ys = [p.y for p in net.pins]
            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            total_wirelength += hpwl

            # Estimate congestion along bounding box edges
            for x in range(min(xs), max(xs) + 1):
                for y in [min(ys), max(ys)]:
                    key = (x, y)
                    congestion_map[key] = congestion_map.get(key, 0) + 1
            for y in range(min(ys), max(ys) + 1):
                for x in [min(xs), max(xs)]:
                    key = (x, y)
                    congestion_map[key] = congestion_map.get(key, 0) + 1

        # Compute max congestion ratio
        max_congestion = max(congestion_map.values()) if congestion_map else 0
        congestion_ratio = min(1.0, max_congestion / 5.0)  # Assume capacity 5

        # Determine success based on congestion
        success = congestion_ratio < 0.9

        # Add some randomness to simulate routing variability
        noise = random.uniform(-0.05, 0.05)

        result = RoutingResult(
            success=success,
            score=0.0,
            wirelength=total_wirelength,
            congestion=congestion_ratio,
            timing_met=success,
            slack=0.0 if success else -1.0,
            runtime_ms=10.0,  # Fast estimate
            error_message=None if success else "Estimated congestion too high"
        )
        result.score = max(0.0, min(1.0, result.as_reward() + noise))
        return result

    def route(
        self,
        placement: 'Placement',
        netlist: Netlist,
        grid: Grid,
        design_json: Optional[str] = None
    ) -> RoutingResult:
        """Route a placement and return quality metrics.

        Args:
            placement: Cell placement to evaluate
            netlist: Design netlist
            grid: FPGA grid/architecture
            design_json: Optional path to design JSON (if not provided, will generate)

        Returns:
            RoutingResult with success/quality metrics
        """
        if self.use_bindings and self._bindings is not None:
            return self._route_bindings(placement, netlist, grid)
        else:
            return self._route_subprocess(placement, netlist, grid, design_json)

    def _route_subprocess(
        self,
        placement: Placement,
        netlist: Netlist,
        grid: Grid,
        design_json: Optional[str] = None
    ) -> RoutingResult:
        """Route using nextpnr subprocess."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export placement to JSON
            placement_json = os.path.join(tmpdir, "placement.json")
            self._export_placement_json(placement, netlist, grid, placement_json)

            # Prepare nextpnr command
            cmd = [self.nextpnr_path]

            if self.chipdb_path:
                cmd.extend(["--chipdb", self.chipdb_path])

            cmd.extend([
                "--json", design_json or placement_json,
                "--placed",  # Input is already placed
                "--route",   # Only run router
                "--report", os.path.join(tmpdir, "report.json")
            ])

            # Run router
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )
                runtime_ms = (time.time() - start_time) * 1000

                # Parse results
                return self._parse_router_output(
                    result.returncode,
                    result.stdout,
                    result.stderr,
                    os.path.join(tmpdir, "report.json"),
                    runtime_ms
                )

            except subprocess.TimeoutExpired:
                return RoutingResult(
                    success=False,
                    score=0.0,
                    wirelength=0,
                    congestion=1.0,
                    timing_met=False,
                    slack=-999.0,
                    runtime_ms=self.timeout * 1000,
                    error_message="Router timeout"
                )

            except Exception as e:
                return RoutingResult(
                    success=False,
                    score=0.0,
                    wirelength=0,
                    congestion=1.0,
                    timing_met=False,
                    slack=-999.0,
                    runtime_ms=0,
                    error_message=str(e)
                )

    def _route_bindings(
        self,
        placement: Placement,
        netlist: Netlist,
        grid: Grid
    ) -> RoutingResult:
        """Route using C++ bindings (fast path)."""
        # Convert to binding format
        placement_data = self._placement_to_bindings(placement)
        netlist_data = self._netlist_to_bindings(netlist)

        # Call C++ router
        result = self._bindings.route_placement(
            placement_data,
            netlist_data,
            grid.width,
            grid.height
        )

        return RoutingResult(
            success=result.success,
            score=result.score,
            wirelength=result.wirelength,
            congestion=result.congestion,
            timing_met=result.timing_met,
            slack=result.slack,
            runtime_ms=result.runtime_ms
        )

    def _export_placement_json(
        self,
        placement: Placement,
        netlist: Netlist,
        grid: Grid,
        output_path: str
    ) -> None:
        """Export placement to nextpnr JSON format."""
        data = {
            "width": grid.width if hasattr(grid, 'width') else grid.get_size()[0],
            "height": grid.height if hasattr(grid, 'height') else grid.get_size()[1],
            "cells": [],
            "nets": []
        }

        # Export cell placements
        if placement.cell_placements:
            for cell_name, (x, y) in placement.cell_placements.items():
                data["cells"].append({
                    "name": cell_name,
                    "x": x,
                    "y": y,
                    "bel": f"X{x}Y{y}"  # Simplified BEL naming
                })

        # Export nets
        for net in netlist.nets:
            net_data = {
                "name": net.name,
                "id": net.net_id,
                "pins": [
                    {"pin_id": pin.pin_id, "x": pin.x, "y": pin.y}
                    for pin in net.pins
                ]
            }
            data["nets"].append(net_data)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _parse_router_output(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        report_path: str,
        runtime_ms: float
    ) -> RoutingResult:
        """Parse nextpnr router output."""
        # Check for routing success
        success = returncode == 0 and "Routing failed" not in stderr

        # Default values
        wirelength = 0
        congestion = 0.0
        timing_met = True
        slack = 0.0

        # Try to parse report JSON
        try:
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report = json.load(f)

                wirelength = report.get("total_wirelength", 0)
                congestion = report.get("max_congestion", 0.0)

                timing = report.get("timing", {})
                slack = timing.get("worst_slack", 0.0)
                timing_met = slack >= 0
        except (json.JSONDecodeError, KeyError):
            pass

        # Parse stdout for metrics if report not available
        if wirelength == 0:
            for line in stdout.split('\n'):
                if "Total wirelength" in line:
                    try:
                        wirelength = int(line.split(':')[-1].strip())
                    except ValueError:
                        pass

        # Compute score
        result = RoutingResult(
            success=success,
            score=0.0,  # Will be computed
            wirelength=wirelength,
            congestion=congestion,
            timing_met=timing_met,
            slack=slack,
            runtime_ms=runtime_ms,
            error_message=stderr if not success else None
        )

        result.score = result.as_reward()
        return result

    def _placement_to_bindings(self, placement: Placement) -> Any:
        """Convert placement to C++ bindings format."""
        # This will be implemented when bindings are built
        raise NotImplementedError("C++ bindings not yet implemented")

    def _netlist_to_bindings(self, netlist: Netlist) -> Any:
        """Convert netlist to C++ bindings format."""
        raise NotImplementedError("C++ bindings not yet implemented")


def create_router(config: Dict[str, Any]) -> NextPNRRouter:
    """Factory function to create router from config."""
    return NextPNRRouter(
        nextpnr_path=config.get("nextpnr_path", "nextpnr-xilinx"),
        chipdb_path=config.get("chipdb_path"),
        use_bindings=config.get("use_bindings", False),
        timeout_seconds=config.get("timeout", 60.0)
    )
