"""Composite reward function."""

from typing import Dict
from ..routing.state import RoutingState
from ..routing.constraints import Constraints
from .metrics import (
    compute_wirelength,
    compute_via_count,
    compute_drc_violations,
    compute_congestion
)


class RewardFunction:
    """Composite reward function.
    
    R = -α·WL - β·vias - γ·DRC - δ·congestion
    """
    
    def __init__(
        self,
        wirelength_weight: float = 1.0,
        via_weight: float = 1.0,
        drc_weight: float = 10.0,
        congestion_weight: float = 1.0
    ):
        self.wirelength_weight = wirelength_weight
        self.via_weight = via_weight
        self.drc_weight = drc_weight
        self.congestion_weight = congestion_weight
    
    def compute(
        self,
        state: RoutingState,
        constraints: Constraints
    ) -> float:
        """Compute reward for routing state.
        
        Args:
            state: Routing state
            constraints: Routing constraints
        
        Returns:
            Reward value (negative, higher is better)
        """
        wirelength = compute_wirelength(state)
        vias = compute_via_count(state)
        drc_violations = compute_drc_violations(state, constraints)
        congestion = compute_congestion(state, constraints)
        
        reward = (
            -self.wirelength_weight * wirelength
            -self.via_weight * vias
            -self.drc_weight * drc_violations
            -self.congestion_weight * congestion
        )
        
        return reward
    
    def compute_metrics(
        self,
        state: RoutingState,
        constraints: Constraints
    ) -> Dict[str, float]:
        """Compute all metrics separately.
        
        Returns:
            Dictionary of metric values
        """
        return {
            "wirelength": compute_wirelength(state),
            "vias": compute_via_count(state),
            "drc_violations": compute_drc_violations(state, constraints),
            "congestion": compute_congestion(state, constraints),
            "reward": self.compute(state, constraints)
        }

