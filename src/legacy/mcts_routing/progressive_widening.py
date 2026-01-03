"""Progressive widening logic."""

from .node import MCTSNode


def should_expand(
    node: MCTSNode,
    num_children: int,
    k: float = 2.0,
    alpha: float = 0.5
) -> bool:
    """Check if node should be expanded (progressive widening).
    
    Args:
        node: Node to check
        num_children: Current number of children
        k: Base constant
        alpha: Exponent
    
    Returns:
        True if should expand
    """
    max_children = max_children_count(node, k, alpha)
    return num_children < max_children


def max_children_count(
    node: MCTSNode,
    k: float = 2.0,
    alpha: float = 0.5
) -> int:
    """Compute maximum number of children for progressive widening.
    
    Formula: max_children = k * N^alpha
    
    Args:
        node: Node
        k: Base constant
        alpha: Exponent
    
    Returns:
        Maximum number of children
    """
    n = node.visit_count
    max_children = int(k * (n ** alpha))
    return max(1, max_children)  # At least 1

