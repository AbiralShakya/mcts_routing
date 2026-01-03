"""Value backpropagation with time-aware normalization."""

from typing import Optional
from .node import MCTSNode
from .tree import MCTSTree
from .value_normalization import normalize_q_value
from ..diffusion.schedule import NoiseSchedule


def backpropagate(
    node: MCTSNode,
    reward: float,
    tree: MCTSTree,
    schedule: Optional[NoiseSchedule] = None,
    T: int = 1000,
    time_aware: bool = True,
    use_proxy: bool = False,
    proxy_weight: float = 0.3
) -> None:
    """Backpropagate reward through tree with time-aware normalization.
    
    Args:
        node: Node to backpropagate from
        reward: Reward value
        schedule: Noise schedule (for time-aware normalization)
        T: Total timesteps
        time_aware: Whether to use time-aware normalization
        use_proxy: Whether reward is a proxy reward
        proxy_weight: Weight for proxy rewards
    """
    current = node
    path = []
    
    # Collect path from node to root
    while current is not None:
        path.append(current)
        # Find parent (simplified - in practice need parent pointers)
        current = None  # TODO: Implement parent tracking
    
    # Backpropagate along path
    for node in path:
        # Update visit count
        node.visit_count += 1
        
        # Normalize reward if time-aware
        if time_aware and schedule is not None:
            normalized_reward = normalize_q_value(
                reward,
                node.timestep,
                schedule,
                T
            )
        else:
            normalized_reward = reward
        
        # Update Q-value (running average)
        # Q = (Q * (N-1) + reward) / N
        if node.visit_count == 1:
            node.q_value = normalized_reward
        else:
            node.q_value = (
                node.q_value * (node.visit_count - 1) + normalized_reward
            ) / node.visit_count


def update_q_value(
    node: MCTSNode,
    reward: float,
    normalized: bool = False
) -> None:
    """Update Q-value of a node (running average).
    
    Args:
        node: Node to update
        reward: Reward value
        normalized: Whether reward is already normalized
    """
    # Note: This won't work with frozen dataclass
    # Q-values should be updated via tree structure
    # This is a placeholder interface
    pass

