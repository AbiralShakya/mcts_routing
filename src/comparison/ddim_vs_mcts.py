"""DDIM vs MCTS comparison framework."""

from typing import Dict, Any, List
import numpy as np


def compare_methods(
    methods: List[str],
    config: Dict[str, Any],
    num_runs: int = 30
) -> Dict[str, Any]:
    """Compare different methods with compute parity.
    
    Args:
        methods: List of method names
        config: Configuration
        num_runs: Number of runs for statistical significance
    
    Returns:
        Comparison results
    """
    # TODO: Implement comparison framework
    # Track compute (UNet calls or wall-clock time)
    # Run each method with same compute budget
    # Collect rewards and compute statistics
    return {"results": {}}

