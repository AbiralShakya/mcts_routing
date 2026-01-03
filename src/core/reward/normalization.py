"""Reward normalization utilities."""

from typing import Optional
import numpy as np


def normalize_reward(
    reward: float,
    min_reward: Optional[float] = None,
    max_reward: Optional[float] = None,
    target_min: float = -1.0,
    target_max: float = 1.0
) -> float:
    """Normalize reward to [target_min, target_max] range.
    
    Args:
        reward: Raw reward value
        min_reward: Minimum reward value (for normalization)
        max_reward: Maximum reward value (for normalization)
        target_min: Target minimum value
        target_max: Target maximum value
    
    Returns:
        Normalized reward
    """
    if min_reward is None or max_reward is None:
        return reward  # Cannot normalize without bounds
    
    if max_reward == min_reward:
        return (target_min + target_max) / 2.0
    
    normalized = (reward - min_reward) / (max_reward - min_reward)
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def compute_reward_statistics(rewards: list) -> dict:
    """Compute reward statistics.
    
    Args:
        rewards: List of reward values
    
    Returns:
        Dictionary with mean, std, min, max
    """
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    rewards_array = np.array(rewards)
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array))
    }

