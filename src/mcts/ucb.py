"""UCB selection for routing MCTS."""

import math
from typing import List, Optional
from .node import RoutingNode


def ucb_score(
    child: RoutingNode,
    parent: RoutingNode,
    c: float = 1.41
) -> float:
    """Compute UCB score.

    UCB = Q + c * sqrt(ln(N_parent) / N_child)

    Args:
        child: Child node
        parent: Parent node
        c: Exploration constant

    Returns:
        UCB score
    """
    if child.visit_count == 0:
        return float('inf')

    exploitation = child.Q
    exploration = c * math.sqrt(
        math.log(parent.visit_count + 1) / child.visit_count
    )

    return exploitation + exploration


def ucb_select(
    root: RoutingNode,
    c: float = 1.41
) -> RoutingNode:
    """Select node using UCB.

    Traverse from root, selecting highest UCB child,
    until reaching a leaf.

    Args:
        root: Root node
        c: Exploration constant

    Returns:
        Selected leaf node
    """
    node = root

    while node.children:
        # Filter non-pruned children
        valid = [child for child in node.children if not child.pruned]

        if not valid:
            break

        # Select best UCB
        node = max(valid, key=lambda child: ucb_score(child, node, c))

    return node


def select_most_visited(node: RoutingNode) -> Optional[RoutingNode]:
    """Select most visited child (for final selection)."""
    valid = [child for child in node.children if not child.pruned]

    if not valid:
        return None

    return max(valid, key=lambda child: child.visit_count)
