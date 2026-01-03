"""MCTS module: Tree search for routing.

This is Monte Carlo Tree Search with a diffusion policy.
Diffusion proposes routing decisions.
The critic stops wasting compute.
The real router provides ground-truth reward.
The tree remembers what worked.
"""

from .node import RoutingNode
from .tree import MCTSTree
from .ucb import ucb_select, ucb_score
from .backprop import backpropagate
from .search import MCTSRouter, iterate

__all__ = [
    "RoutingNode",
    "MCTSTree",
    "ucb_select",
    "ucb_score",
    "backpropagate",
    "MCTSRouter",
    "iterate"
]
