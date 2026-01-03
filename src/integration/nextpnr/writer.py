"""nextpnr writer: routes → nextpnr format/JSON."""

import json
from typing import List, Dict
from ...core.routing.state import RoutingState


class NextPNRWriter:
    """Writer for nextpnr input formats."""
    
    def write_routes(
        self,
        routing_state: RoutingState,
        file_path: str,
        format: str = "json"
    ) -> None:
        """Write routes to nextpnr format.
        
        Args:
            routing_state: Routing state to write
            file_path: Output file path
            format: Output format ("json" or "nextpnr")
        """
        if format == "json":
            self._write_json(routing_state, file_path)
        elif format == "nextpnr":
            self._write_nextpnr_format(routing_state, file_path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _write_json(
        self,
        routing_state: RoutingState,
        file_path: str
    ) -> None:
        """Write routes in JSON format.
        
        Args:
            routing_state: Routing state
            file_path: Output file path
        """
        routes = []
        
        # Group wires by net
        net_wires: Dict[int, List] = {}
        for wire in routing_state.wires:
            if wire.net_id not in net_wires:
                net_wires[wire.net_id] = []
            net_wires[wire.net_id].append(wire)
        
        # Convert to JSON format
        for net_id, wires in net_wires.items():
            segments = []
            for wire in wires:
                segments.append({
                    "start": [wire.start[0], wire.start[1]],
                    "end": [wire.end[0], wire.end[1]],
                    "layer": wire.layer
                })
            
            routes.append({
                "net_id": net_id,
                "segments": segments
            })
        
        data = {
            "routes": routes,
            "grid": {
                "width": routing_state.grid.width,
                "height": routing_state.grid.height
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _write_nextpnr_format(
        self,
        routing_state: RoutingState,
        file_path: str
    ) -> None:
        """Write routes in nextpnr's native format.
        
        Args:
            routing_state: Routing state
            file_path: Output file path
        """
        # nextpnr uses a specific format for routing
        # This is a simplified version - actual format may vary
        lines = []
        lines.append("# nextpnr routing file")
        lines.append(f"# Grid: {routing_state.grid.width}x{routing_state.grid.height}")
        lines.append("")
        
        # Group wires by net
        net_wires: Dict[int, List] = {}
        for wire in routing_state.wires:
            if wire.net_id not in net_wires:
                net_wires[wire.net_id] = []
            net_wires[wire.net_id].append(wire)
        
        # Write each net
        for net_id, wires in net_wires.items():
            lines.append(f"net {net_id}")
            for wire in wires:
                lines.append(
                    f"  wire ({wire.start[0]},{wire.start[1]}) -> ({wire.end[0]},{wire.end[1]})"
                )
            lines.append("")
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))


def write_routes(routing_state: RoutingState, file_path: str) -> None:
    """Convenience function to write routes."""
    writer = NextPNRWriter()
    writer.write_routes(routing_state, file_path, format="json")
