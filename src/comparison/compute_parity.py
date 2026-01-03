"""Compute parity tracking: count UNet forward passes for fair comparison."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ComputeTracker:
    """Track compute usage (UNet calls, wall-clock time)."""
    
    unet_calls: int = 0
    wall_clock_time: float = 0.0
    method_name: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def reset(self):
        """Reset counters."""
        self.unet_calls = 0
        self.wall_clock_time = 0.0
        self.metadata = {}
    
    def add_unet_call(self, count: int = 1):
        """Add UNet forward pass."""
        self.unet_calls += count
    
    def add_time(self, seconds: float):
        """Add wall-clock time."""
        self.wall_clock_time += seconds
    
    def get_summary(self) -> Dict:
        """Get summary of compute usage."""
        return {
            'method': self.method_name,
            'unet_calls': self.unet_calls,
            'wall_clock_time': self.wall_clock_time,
            'metadata': self.metadata
        }


class ComputeParityManager:
    """Manage compute parity across different methods."""
    
    def __init__(self, target_unet_calls: Optional[int] = None):
        """Initialize compute parity manager.
        
        Args:
            target_unet_calls: Target number of UNet calls (if None, uses first method's count)
        """
        self.target_unet_calls = target_unet_calls
        self.trackers: Dict[str, ComputeTracker] = {}
    
    def create_tracker(self, method_name: str) -> ComputeTracker:
        """Create a compute tracker for a method.
        
        Args:
            method_name: Name of the method
        
        Returns:
            ComputeTracker
        """
        tracker = ComputeTracker(method_name=method_name)
        self.trackers[method_name] = tracker
        return tracker
    
    def get_tracker(self, method_name: str) -> Optional[ComputeTracker]:
        """Get tracker for a method.
        
        Args:
            method_name: Name of the method
        
        Returns:
            ComputeTracker or None
        """
        return self.trackers.get(method_name)
    
    def check_parity(self) -> Dict[str, bool]:
        """Check if all methods have similar compute usage.
        
        Returns:
            Dict mapping method name to whether it's within parity
        """
        if not self.trackers:
            return {}
        
        # Use target or first method's count
        if self.target_unet_calls is None:
            first_tracker = list(self.trackers.values())[0]
            target = first_tracker.unet_calls
        else:
            target = self.target_unet_calls
        
        # Allow 10% variance
        tolerance = target * 0.1
        
        results = {}
        for name, tracker in self.trackers.items():
            diff = abs(tracker.unet_calls - target)
            results[name] = diff <= tolerance
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary of all trackers."""
        summary = {}
        for name, tracker in self.trackers.items():
            summary[name] = tracker.get_summary()
        
        parity = self.check_parity()
        summary['parity_check'] = parity
        
        return summary


def count_unet_calls_ddpm(T: int) -> int:
    """Count UNet calls for DDPM.
    
    Args:
        T: Number of timesteps
    
    Returns:
        Number of UNet calls
    """
    return T  # One call per timestep


def count_unet_calls_ddim(T: int, step_size: int = 1) -> int:
    """Count UNet calls for DDIM.
    
    Args:
        T: Number of timesteps
        step_size: Step size (default 1, can be larger for faster sampling)
    
    Returns:
        Number of UNet calls
    """
    return T // step_size


def count_unet_calls_mcts(
    num_iterations: int,
    avg_rollout_depth: int
) -> int:
    """Count UNet calls for MCTS.
    
    Args:
        num_iterations: Number of MCTS iterations
        avg_rollout_depth: Average rollout depth
    
    Returns:
        Number of UNet calls
    """
    # Each iteration: selection (tree traversal) + expansion + rollout
    # Selection: ~log(depth) calls (negligible)
    # Expansion: 1 call per new node
    # Rollout: avg_rollout_depth calls
    return num_iterations * (1 + avg_rollout_depth)


def count_unet_calls_hybrid(
    T: int,
    ddim_steps: int,
    mcts_iterations: int,
    avg_rollout_depth: int
) -> int:
    """Count UNet calls for hybrid DDIM-MCTS.
    
    Args:
        T: Total timesteps
        ddim_steps: Number of DDIM steps (first phase)
        mcts_iterations: Number of MCTS iterations (second phase)
        avg_rollout_depth: Average MCTS rollout depth
    
    Returns:
        Number of UNet calls
    """
    ddim_calls = ddim_steps
    mcts_calls = count_unet_calls_mcts(mcts_iterations, avg_rollout_depth)
    return ddim_calls + mcts_calls

