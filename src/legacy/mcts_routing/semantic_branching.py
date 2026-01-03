"""Semantic branching control (domain knowledge injection).

Note: This is domain knowledge injection - must be ablated.
"""

from typing import Optional
import torch

from .node import MCTSNode
from ..decoding.potential_decoder import PotentialDecoder
from ..routing.grid import Grid
from ..routing.netlist import Netlist
from ..routing.state import RoutingState


def should_branch_semantically(
    node: MCTSNode,
    grid: Grid,
    netlist: Netlist,
    decoder: Optional[PotentialDecoder] = None,
    routing_state: Optional[RoutingState] = None,
    enabled: bool = True,
    branch_on_unresolved_nets: bool = True,
    branch_on_congestion: bool = True,
    branch_on_entropy: bool = True,
    congestion_threshold: float = 0.7,
    entropy_threshold: float = 0.5
) -> bool:
    """Determine if should branch based on semantic signals.
    
    Args:
        node: Current node
        grid: Grid structure
        netlist: Netlist
        decoder: Potential decoder
        routing_state: Current routing state (if available)
        enabled: Whether semantic branching is enabled
        branch_on_unresolved_nets: Branch if unresolved nets exist
        branch_on_congestion: Branch if high congestion
        branch_on_entropy: Branch if high decoder entropy
        congestion_threshold: Congestion threshold
        entropy_threshold: Entropy threshold
    
    Returns:
        True if should branch semantically
    """
    if not enabled:
        return False  # Uniform branching
    
    # Check unresolved nets
    if branch_on_unresolved_nets and routing_state is not None:
        unresolved = count_unresolved_nets(routing_state, netlist)
        if unresolved > 0:
            return True
    
    # Check congestion
    if branch_on_congestion and routing_state is not None:
        from ..routing.constraints import Constraints
        constraints = Constraints()
        congestion = constraints.get_congestion(routing_state, grid)
        if congestion > congestion_threshold:
            return True
    
    # Check decoder entropy
    if branch_on_entropy and decoder is not None:
        from .value_bootstrapping import compute_congestion_entropy
        try:
            potentials = decoder.decode(node.latent, grid, netlist)
            entropy = compute_congestion_entropy(potentials)
            if entropy < entropy_threshold:  # Low entropy = high concentration = branch
                return True
        except:
            pass  # If decoding fails, don't branch
    
    return False


def count_unresolved_nets(
    routing_state: RoutingState,
    netlist: Netlist
) -> int:
    """Count number of unresolved nets.
    
    Args:
        routing_state: Current routing state
        netlist: Netlist
    
    Returns:
        Number of unresolved nets
    """
    unresolved = 0
    for net in netlist.nets:
        if not routing_state.is_net_routed(net.net_id):
            unresolved += 1
    return unresolved

