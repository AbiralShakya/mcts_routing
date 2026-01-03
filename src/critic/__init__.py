"""Critic module: Early failure detection for routing.

Predicts routing success from partial routing state.
Enables early pruning of doomed MCTS paths.

Key insight: This is NOT a heuristic.
Trained on (partial routing → final nextpnr score) pairs.
Learns global impossibility signals:
- Early congestion collapse
- Irrecoverable net blockages
- Timing dead ends
"""

from .gnn import RoutingCritic
from .features import RoutingGraphBuilder
from .training import CriticTrainer

__all__ = ["RoutingCritic", "RoutingGraphBuilder", "CriticTrainer"]
