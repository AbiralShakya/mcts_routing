"""MCTS node for routing search."""

from dataclasses import dataclass, field
from typing import List, Optional
import torch

from ..diffusion.model import RoutingState


@dataclass
class RoutingNode:
    """MCTS node for routing search.

    State is partial routing assignment - decoded on demand.

    Attributes:
        state: RoutingState with net latents and routed nets
        parent: Parent node (None for root)
        children: List of child nodes
        visit_count: N - number of times visited
        total_value: W - sum of backpropagated rewards
        pruned: Whether critic killed this branch
    """
    state: RoutingState
    parent: Optional['RoutingNode'] = None
    children: List['RoutingNode'] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    pruned: bool = False

    @property
    def Q(self) -> float:
        """Average value (Q = W / N)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def Q_normalized(self) -> float:
        """Time-aware normalized Q-value.

        Scales Q by timestep uncertainty:
        - Early timesteps (high t) = more uncertain = higher exploration bonus
        - Late timesteps (low t) = more committed = trust Q more

        This helps UCB balance exploration across tree depths.
        """
        base_Q = self.Q
        max_timesteps = 1000  # Assume standard diffusion schedule

        # Time scale: 0 at t=0, 1 at t=max_timesteps
        time_scale = self.t / max_timesteps

        # Uncertainty multiplier: earlier = more uncertain
        # sqrt scaling for smoother transition
        uncertainty = 1.0 + 0.5 * (time_scale ** 0.5)

        return base_Q * uncertainty

    @property
    def t(self) -> int:
        """Shorthand for timestep."""
        return self.state.timestep

    def is_terminal(self) -> bool:
        """Check if node is terminal (t = 0 or all nets routed)."""
        return self.state.is_terminal()

    def is_leaf(self) -> bool:
        """Check if node has no children."""
        return len(self.children) == 0

    def add_child(self, child: 'RoutingNode') -> None:
        """Add child node."""
        child.parent = self
        self.children.append(child)

    def get_unrouted_count(self) -> int:
        """Number of nets still unrouted."""
        return len(self.state.get_unrouted_nets())

    def __repr__(self) -> str:
        return (f"RoutingNode(t={self.t}, visits={self.visit_count}, "
                f"Q={self.Q:.3f}, unrouted={self.get_unrouted_count()})")
