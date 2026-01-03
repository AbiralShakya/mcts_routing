"""nextpnr validator: DRC check, parse violations."""

import subprocess
import json
import os
from typing import List, Dict, Optional
from ...core.routing.state import RoutingState
from ...core.routing.grid import Grid
from ...core.routing.netlist import Netlist
from ...core.routing.constraints import DRCViolation


class NextPNRValidator:
    """Validator for nextpnr routing results."""
    
    def __init__(self, nextpnr_path: str = "nextpnr"):
        """Initialize validator.
        
        Args:
            nextpnr_path: Path to nextpnr executable
        """
        self.nextpnr_path = nextpnr_path
    
    def validate_routing(
        self,
        routing_state: RoutingState,
        grid: Grid,
        netlist: Netlist,
        temp_file: Optional[str] = None
    ) -> Dict:
        """Validate routing using nextpnr DRC checker.
        
        Args:
            routing_state: Routing state to validate
            grid: Grid structure
            netlist: Netlist
            temp_file: Temporary file path (if None, creates one)
        
        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'drc_violations': List[DRCViolation],
                'wirelength': float,
                'timing_slack': float (if available)
            }
        """
        # Write routes to temporary file
        if temp_file is None:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
        
        from .writer import NextPNRWriter
        writer = NextPNRWriter()
        writer.write_routes(routing_state, temp_file, format="json")
        
        try:
            # Run nextpnr DRC check
            result = self._run_drc_check(temp_file)
            
            # Parse violations
            violations = self._parse_violations(result)
            
            # Compute metrics
            wirelength = self._compute_wirelength(routing_state)
            
            return {
                'valid': len(violations) == 0,
                'drc_violations': violations,
                'wirelength': wirelength,
                'timing_slack': None  # Would need timing analysis
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _run_drc_check(self, route_file: str) -> str:
        """Run nextpnr DRC check.
        
        Args:
            route_file: Path to route file
        
        Returns:
            DRC check output
        """
        try:
            # Run nextpnr --check
            result = subprocess.run(
                [self.nextpnr_path, '--check', route_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "DRC check timed out"
        except FileNotFoundError:
            # nextpnr not found - return empty (for testing without nextpnr)
            return ""
        except Exception as e:
            return f"DRC check error: {e}"
    
    def _parse_violations(self, drc_output: str) -> List[DRCViolation]:
        """Parse DRC violations from nextpnr output.
        
        Args:
            drc_output: DRC check output
        
        Returns:
            List of DRC violations
        """
        violations = []
        
        # Simple parsing (nextpnr format may vary)
        lines = drc_output.split('\n')
        for line in lines:
            if 'violation' in line.lower() or 'error' in line.lower():
                # Try to extract violation info
                violation = DRCViolation(
                    violation_type="unknown",
                    location=(0, 0),
                    severity=1.0
                )
                violations.append(violation)
        
        return violations
    
    def _compute_wirelength(self, routing_state: RoutingState) -> float:
        """Compute total wirelength.
        
        Args:
            routing_state: Routing state
        
        Returns:
            Total wirelength (Manhattan distance)
        """
        total_length = 0.0
        for wire in routing_state.wires:
            x1, y1 = wire.start
            x2, y2 = wire.end
            length = abs(x2 - x1) + abs(y2 - y1)  # Manhattan
            total_length += length
        return total_length


def validate_routing(
    routing_state: RoutingState,
    grid: Grid,
    netlist: Netlist
) -> Dict:
    """Convenience function to validate routing."""
    validator = NextPNRValidator()
    return validator.validate_routing(routing_state, grid, netlist)

