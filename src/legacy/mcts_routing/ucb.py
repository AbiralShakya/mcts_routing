"""UCB selection policy."""

from typing import List, Optional
from .node import MCTSNode


def ucb_score(
    parent: MCTSNode,
    child: MCTSNode,
    c: float = 1.41,
    q_normalized: Optional[float] = None
) -> float:
    """Compute UCB score for child node.
    
    Args:
        parent: Parent node
        child: Child node
        c: Exploration constant
        q_normalized: Normalized Q-value (if time-aware normalization applied)
    
    Returns:
        UCB score
    """
    if child.visit_count == 0:
        return float('inf')  # Unvisited nodes get infinite score
    
    # Use normalized Q if provided, otherwise use raw Q
    q = q_normalized if q_normalized is not None else child.q_value
    
    # UCB formula: Q + c * sqrt(log(N) / n)
    exploration = c * (parent.visit_count ** 0.5) / (1 + child.visit_count)
    return q + exploration


def select_best_child(
    node: MCTSNode,
    children: List[MCTSNode],
    c: float = 1.41,
    q_normalized_map: Optional[dict] = None
) -> MCTSNode:
    """Select best child according to UCB.
    
    Args:
        node: Parent node
        children: List of child nodes
        c: Exploration constant
        q_normalized_map: Map from child to normalized Q-value
    
    Returns:
        Best child node
    """
    if not children:
        return None
    
    best_child = None
    best_score = float('-inf')
    
    for child in children:
        q_normalized = q_normalized_map.get(id(child)) if q_normalized_map else None
        score = ucb_score(node, child, c, q_normalized)
        if score > best_score:
            best_score = score
            best_child = child
    
    return best_child

