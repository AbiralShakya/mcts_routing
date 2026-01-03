"""Backpropagation for routing MCTS.

Each rollout updates:
- Visit counts
- Mean reward
- Exploration statistics

Good routing decisions reinforce early denoising steps.
"""

from typing import List
from .node import RoutingNode


def backpropagate(node: RoutingNode, reward: float) -> None:
    """Backpropagate reward from node to root.

    Updates visit_count and total_value for all ancestors.

    Args:
        node: Terminal or pruned node
        reward: Reward value (0 if pruned, actual score if terminal)
    """
    current = node

    while current is not None:
        current.visit_count += 1
        current.total_value += reward
        current = current.parent


def backpropagate_path(path: List[RoutingNode], reward: float) -> None:
    """Backpropagate along explicit path."""
    for node in path:
        node.visit_count += 1
        node.total_value += reward
