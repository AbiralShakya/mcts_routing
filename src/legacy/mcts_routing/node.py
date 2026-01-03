"""MCTS node representation.

Note: State is (x_t, t) ONLY. Routing state is NOT stored here.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class MCTSNode:
    """MCTS node with state (x_t, t) only.
    
    Routing state is derived, never stored in node.
    Note: Not frozen to allow Q-value updates via Tree class.
    """
    latent: torch.Tensor  # x_t
    timestep: int  # t
    visit_count: int = 0
    q_value: float = 0.0
    
    def __hash__(self) -> int:
        """Hash based on latent and timestep only."""
        # Hash latent tensor and timestep
        latent_hash = hash(tuple(self.latent.flatten().tolist()))
        return hash((latent_hash, self.timestep))
    
    def __eq__(self, other) -> bool:
        """Equality based on latent and timestep."""
        if not isinstance(other, MCTSNode):
            return False
        return (
            torch.allclose(self.latent, other.latent, atol=1e-5) and
            self.timestep == other.timestep
        )
    
    def is_terminal(self, min_timestep: int = 0) -> bool:
        """Check if node is terminal (t = min_timestep)."""
        return self.timestep <= min_timestep

