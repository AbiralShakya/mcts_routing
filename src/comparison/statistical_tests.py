"""Statistical significance testing for MCTS vs baselines."""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional


def welch_ttest(
    mcts_rewards: List[float],
    baseline_rewards: List[float],
    alternative: str = "greater"
) -> Tuple[float, float, bool]:
    """Welch's t-test for unequal variances.
    
    H_0: μ_MCTS = μ_baseline
    H_1: μ_MCTS > μ_baseline (or <, or !=)
    
    Args:
        mcts_rewards: List of MCTS rewards
        baseline_rewards: List of baseline rewards
        alternative: "greater", "less", or "two-sided"
    
    Returns:
        (t_statistic, p_value, significant)
        where significant is True if p < 0.05
    """
    mcts_array = np.array(mcts_rewards)
    baseline_array = np.array(baseline_rewards)
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(
        mcts_array,
        baseline_array,
        equal_var=False,
        alternative=alternative
    )
    
    significant = p_value < 0.05
    
    return float(t_stat), float(p_value), significant


def mann_whitney_u_test(
    mcts_rewards: List[float],
    baseline_rewards: List[float],
    alternative: str = "greater"
) -> Tuple[float, float, bool]:
    """Mann-Whitney U test (non-parametric).
    
    Useful when distributions are not normal.
    
    Args:
        mcts_rewards: List of MCTS rewards
        baseline_rewards: List of baseline rewards
        alternative: "greater", "less", or "two-sided"
    
    Returns:
        (u_statistic, p_value, significant)
    """
    u_stat, p_value = stats.mannwhitneyu(
        mcts_rewards,
        baseline_rewards,
        alternative=alternative
    )
    
    significant = p_value < 0.05
    
    return float(u_stat), float(p_value), significant


def compute_effect_size(
    mcts_rewards: List[float],
    baseline_rewards: List[float]
) -> float:
    """Compute Cohen's d (effect size).
    
    d = (μ_MCTS - μ_baseline) / σ_pooled
    
    Args:
        mcts_rewards: List of MCTS rewards
        baseline_rewards: List of baseline rewards
    
    Returns:
        Cohen's d
    """
    mcts_array = np.array(mcts_rewards)
    baseline_array = np.array(baseline_rewards)
    
    mcts_mean = np.mean(mcts_array)
    baseline_mean = np.mean(baseline_array)
    
    mcts_std = np.std(mcts_array, ddof=1)
    baseline_std = np.std(baseline_array, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(mcts_array), len(baseline_array)
    pooled_std = np.sqrt(
        ((n1 - 1) * mcts_std**2 + (n2 - 1) * baseline_std**2) / (n1 + n2 - 2)
    )
    
    if pooled_std > 1e-8:
        d = (mcts_mean - baseline_mean) / pooled_std
        return float(d)
    return 0.0


def statistical_significance_test(
    mcts_rewards: List[float],
    baseline_rewards: List[float],
    min_samples: int = 30,
    test_type: str = "welch"
) -> Dict:
    """Complete statistical significance test.
    
    Args:
        mcts_rewards: List of MCTS rewards
        baseline_rewards: List of baseline rewards
        min_samples: Minimum samples required
        test_type: "welch" or "mannwhitney"
    
    Returns:
        Dict with test results:
        {
            'significant': bool,
            'p_value': float,
            'effect_size': float,
            'mcts_mean': float,
            'baseline_mean': float,
            'mcts_std': float,
            'baseline_std': float
        }
    """
    if len(mcts_rewards) < min_samples or len(baseline_rewards) < min_samples:
        return {
            'significant': False,
            'p_value': 1.0,
            'effect_size': 0.0,
            'error': f'Insufficient samples: {len(mcts_rewards)}, {len(baseline_rewards)}'
        }
    
    # Run appropriate test
    if test_type == "welch":
        _, p_value, significant = welch_ttest(mcts_rewards, baseline_rewards)
    elif test_type == "mannwhitney":
        _, p_value, significant = mann_whitney_u_test(mcts_rewards, baseline_rewards)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Compute effect size
    effect_size = compute_effect_size(mcts_rewards, baseline_rewards)
    
    return {
        'significant': significant,
        'p_value': p_value,
        'effect_size': effect_size,
        'mcts_mean': float(np.mean(mcts_rewards)),
        'baseline_mean': float(np.mean(baseline_rewards)),
        'mcts_std': float(np.std(mcts_rewards)),
        'baseline_std': float(np.std(baseline_rewards)),
        'mcts_n': len(mcts_rewards),
        'baseline_n': len(baseline_rewards)
    }
