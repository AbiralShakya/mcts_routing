"""nextpnr reader: parse JSON/internal format → Grid, Netlist, Placement."""

import json
from typing import Dict, List, Tuple, Optional
from ...core.routing.grid import Grid
from ...core.routing.netlist import Netlist, Net, Pin
from ...core.routing.placement import Placement
from ...core.routing.state import RoutingState, WireSegment


class NextPNRReader:
    """Reader for nextpnr output formats."""
    
    def read_grid(self, file_path: str) -> Grid:
        """Read grid structure from nextpnr JSON.
        
        Args:
            file_path: Path to nextpnr JSON file
        
        Returns:
            Grid structure
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract grid dimensions
        width = data.get('width', 100)
        height = data.get('height', 100)
        
        return Grid(width=width, height=height)
    
    def read_netlist(self, file_path: str) -> Netlist:
        """Read netlist from nextpnr JSON.
        
        Args:
            file_path: Path to nextpnr JSON file
        
        Returns:
            Netlist
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        nets = []
        net_data = data.get('nets', [])
        
        for net_info in net_data:
            net_id = net_info.get('id', len(nets))
            pins_data = net_info.get('pins', [])
            
            pins = []
            for i, pin_data in enumerate(pins_data):
                x = pin_data.get('x', 0)
                y = pin_data.get('y', 0)
                pin_id = pin_data.get('pin_id', i)
                pins.append(Pin(x=x, y=y, pin_id=pin_id))
            
            if len(pins) >= 2:  # Net must have at least 2 pins
                net = Net(net_id=net_id, pins=pins, name=net_info.get('name', f'net_{net_id}'))
                nets.append(net)
        
        return Netlist(nets=nets)
    
    def read_placement(self, file_path: str) -> Placement:
        """Read placement from nextpnr JSON.
        
        Args:
            file_path: Path to nextpnr JSON file
        
        Returns:
            Placement
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        pin_placements = {}
        placement_data = data.get('placement', {})
        pins_data = placement_data.get('pins', [])
        
        for pin_data in pins_data:
            pin_id = pin_data.get('pin_id', len(pin_placements))
            x = pin_data.get('x', 0)
            y = pin_data.get('y', 0)
            pin_placements[pin_id] = (x, y)
        
        cell_placements = {}
        cells_data = placement_data.get('cells', [])
        for cell_data in cells_data:
            cell_name = cell_data.get('name', '')
            x = cell_data.get('x', 0)
            y = cell_data.get('y', 0)
            cell_placements[cell_name] = (x, y)
        
        return Placement(
            pin_placements=pin_placements,
            cell_placements=cell_placements if cell_placements else None
        )
    
    def read_routing(self, file_path: str, grid: Grid, netlist: Netlist) -> RoutingState:
        """Read routing from nextpnr JSON.
        
        Args:
            file_path: Path to nextpnr routing JSON file
            grid: Grid structure
            netlist: Netlist
        
        Returns:
            RoutingState
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        routing_state = RoutingState(grid=grid, netlist=netlist)
        routes_data = data.get('routes', [])
        
        for route_data in routes_data:
            net_id = route_data.get('net_id', -1)
            segments = route_data.get('segments', [])
            
            for seg in segments:
                start = seg.get('start', [0, 0])
                end = seg.get('end', [0, 0])
                layer = seg.get('layer', 0)
                
                wire = WireSegment(
                    start=(start[0], start[1]),
                    end=(end[0], end[1]),
                    net_id=net_id,
                    layer=layer
                )
                routing_state.add_wire(wire)
        
        return routing_state
    
    def read_all(self, file_path: str) -> Tuple[Grid, Netlist, Placement, Optional[RoutingState]]:
        """Read all data from nextpnr JSON file.
        
        Args:
            file_path: Path to nextpnr JSON file
        
        Returns:
            (grid, netlist, placement, routing_state)
        """
        grid = self.read_grid(file_path)
        netlist = self.read_netlist(file_path)
        placement = self.read_placement(file_path)
        
        # Try to read routing if available
        routing_state = None
        try:
            routing_state = self.read_routing(file_path, grid, netlist)
        except (KeyError, ValueError):
            pass  # Routing not available
        
        return grid, netlist, placement, routing_state


def read_nextpnr_placement(file_path: str) -> Placement:
    """Convenience function to read placement."""
    reader = NextPNRReader()
    return reader.read_placement(file_path)


def read_nextpnr_grid(file_path: str) -> Grid:
    """Convenience function to read grid."""
    reader = NextPNRReader()
    return reader.read_grid(file_path)
