"""Bridge module: C++ bindings to nextpnr router.

Provides fast access to nextpnr's router2.cc for scoring placements.
"""

from .router import NextPNRRouter, RoutingResult
from .placement_io import export_placement, import_routing

__all__ = [
    "NextPNRRouter",
    "RoutingResult",
    "export_placement",
    "import_routing"
]
