"""Placement I/O utilities for nextpnr bridge.

Handles conversion between internal placement format and nextpnr formats.
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.routing.placement import Placement
from ..core.routing.netlist import Netlist, Net, Pin
from ..core.routing.grid import Grid
from ..core.routing.state import PhysicalRoutingState, WireSegment


def export_placement(
    placement: Placement,
    netlist: Netlist,
    grid: Grid,
    output_path: str,
    format: str = "json"
) -> None:
    """Export placement to nextpnr-compatible format.

    Args:
        placement: Cell/pin placement
        netlist: Design netlist
        grid: FPGA grid
        output_path: Output file path
        format: Output format ("json" or "pcf")
    """
    if format == "json":
        _export_json(placement, netlist, grid, output_path)
    elif format == "pcf":
        _export_pcf(placement, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _export_json(
    placement: Placement,
    netlist: Netlist,
    grid: Grid,
    output_path: str
) -> None:
    """Export to nextpnr JSON format."""
    width, height = grid.get_size()

    data = {
        "version": "1.0",
        "generator": "mcts_routing",
        "modules": {
            "top": {
                "cells": {},
                "netnames": {}
            }
        },
        "placement": {
            "width": width,
            "height": height,
            "cells": [],
            "pins": []
        }
    }

    # Export cell placements
    if placement.cell_placements:
        for cell_name, (x, y) in placement.cell_placements.items():
            cell_data = {
                "name": cell_name,
                "type": "SLICE",  # Default type
                "x": x,
                "y": y,
                "bel": _coords_to_bel(x, y)
            }
            data["placement"]["cells"].append(cell_data)
            data["modules"]["top"]["cells"][cell_name] = {
                "type": "SLICE",
                "attributes": {"LOC": _coords_to_bel(x, y)}
            }

    # Export pin placements
    for pin_id, (x, y) in placement.pin_placements.items():
        data["placement"]["pins"].append({
            "pin_id": pin_id,
            "x": x,
            "y": y
        })

    # Export nets
    for net in netlist.nets:
        net_data = {
            "name": net.name,
            "driver": net.pins[0].pin_id if net.pins else None,
            "sinks": [p.pin_id for p in net.pins[1:]] if len(net.pins) > 1 else []
        }
        data["modules"]["top"]["netnames"][net.name] = net_data

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _export_pcf(placement: Placement, output_path: str) -> None:
    """Export to PCF (Physical Constraints File) format.

    PCF format:
    set_io <port_name> <pin_location>
    """
    lines = []

    if placement.cell_placements:
        for cell_name, (x, y) in placement.cell_placements.items():
            bel = _coords_to_bel(x, y)
            lines.append(f"set_location {cell_name} {bel}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def _coords_to_bel(x: int, y: int, bel_type: str = "SLICE") -> str:
    """Convert coordinates to BEL name.

    Xilinx naming convention: SLICE_X{x}Y{y}
    """
    return f"{bel_type}_X{x}Y{y}"


def import_routing(
    routing_path: str,
    grid: Grid,
    netlist: Netlist
) -> PhysicalRoutingState:
    """Import routing result from nextpnr.

    Args:
        routing_path: Path to routing JSON file
        grid: FPGA grid
        netlist: Design netlist

    Returns:
        RoutingState with wire segments
    """
    with open(routing_path, 'r') as f:
        data = json.load(f)

    routing_state = PhysicalRoutingState(grid=grid, netlist=netlist)

    # Parse routed nets
    routes = data.get("routes", data.get("routing", {}))

    if isinstance(routes, list):
        for route in routes:
            _parse_route(route, routing_state)
    elif isinstance(routes, dict):
        for net_name, route in routes.items():
            _parse_route(route, routing_state, net_name)

    return routing_state


def _parse_route(
    route_data: Dict[str, Any],
    routing_state: PhysicalRoutingState,
    net_name: Optional[str] = None
) -> None:
    """Parse single net route."""
    net_id = route_data.get("net_id", -1)
    if net_id == -1 and net_name:
        # Try to find net by name
        for net in routing_state.netlist.nets:
            if net.name == net_name:
                net_id = net.net_id
                break

    segments = route_data.get("segments", route_data.get("wires", []))

    for seg in segments:
        if isinstance(seg, dict):
            start = seg.get("start", seg.get("from", [0, 0]))
            end = seg.get("end", seg.get("to", [0, 0]))
            layer = seg.get("layer", 0)
        elif isinstance(seg, list) and len(seg) >= 4:
            start = seg[:2]
            end = seg[2:4]
            layer = seg[4] if len(seg) > 4 else 0
        else:
            continue

        wire = WireSegment(
            start=(start[0], start[1]),
            end=(end[0], end[1]),
            net_id=net_id,
            layer=layer
        )
        routing_state.add_wire(wire)


def parse_nextpnr_log(log_path: str) -> Dict[str, Any]:
    """Parse nextpnr log file for metrics.

    Extracts:
    - Routing success/failure
    - Wirelength
    - Congestion
    - Timing info
    - Runtime
    """
    metrics = {
        "success": True,
        "wirelength": 0,
        "congestion": 0.0,
        "timing_met": True,
        "slack_ns": 0.0,
        "runtime_s": 0.0,
        "iterations": 0
    }

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            if "Routing failed" in line or "FAILED" in line:
                metrics["success"] = False

            elif "Total wirelength" in line:
                try:
                    metrics["wirelength"] = int(line.split(':')[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

            elif "Max congestion" in line:
                try:
                    metrics["congestion"] = float(line.split(':')[-1].strip().rstrip('%')) / 100
                except (ValueError, IndexError):
                    pass

            elif "Slack" in line or "slack" in line:
                try:
                    slack_str = line.split(':')[-1].strip().split()[0]
                    metrics["slack_ns"] = float(slack_str.replace('ns', ''))
                    metrics["timing_met"] = metrics["slack_ns"] >= 0
                except (ValueError, IndexError):
                    pass

            elif "Router iterations" in line:
                try:
                    metrics["iterations"] = int(line.split(':')[-1].strip())
                except (ValueError, IndexError):
                    pass

    return metrics


class PlacementCache:
    """Cache for placement -> routing results.

    Avoids re-running router for same placements during MCTS.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[int, 'RoutingResult'] = {}
        self._access_order: List[int] = []

    def _hash_placement(self, placement: Placement) -> int:
        """Create hash for placement."""
        items = []
        if placement.cell_placements:
            for name in sorted(placement.cell_placements.keys()):
                items.append((name, placement.cell_placements[name]))
        for pin_id in sorted(placement.pin_placements.keys()):
            items.append((pin_id, placement.pin_placements[pin_id]))
        return hash(tuple(items))

    def get(self, placement: Placement) -> Optional['RoutingResult']:
        """Get cached result for placement."""
        key = self._hash_placement(placement)
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, placement: Placement, result: 'RoutingResult') -> None:
        """Cache result for placement."""
        key = self._hash_placement(placement)

        # Evict if full
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = result
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()
