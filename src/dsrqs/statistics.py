# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Advanced statistical analysis for scientific research
#   - Wilcoxon signed-rank tests
#   - Confidence intervals
#   - Effect sizes
#   - Meta-analysis tools
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Dict:
    """
    Calculate confidence interval for mean.
    
    Args:
        data: Array of values
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Dictionary with mean, lower, upper, and confidence level
    """
    n = len(data)
    mean_val = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return {
        "mean": float(mean_val),
        "lower": float(mean_val - h),
        "upper": float(mean_val + h),
        "confidence": confidence,
        "std_err": float(std_err),
        "sample_size": n
    }


def wilcoxon_signed_rank_test(
    data1: np.ndarray, 
    data2: np.ndarray, 
    alternative: str = "two-sided"
) -> Dict:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Args:
        data1: First sample
        data2: Second sample
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
    
    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
    
    # Calculate effect size (r)
    n = len(data1)
    z = stats.norm.ppf(p_value / 2) if alternative == "two-sided" else stats.norm.ppf(p_value)
    r = abs(z) / np.sqrt(n)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_r": float(r),
        "alternative": alternative,
        "sample_size": n,
        "significant": p_value < 0.05
    }


def mann_whitney_u_test(
    data1: np.ndarray, 
    data2: np.ndarray, 
    alternative: str = "two-sided"
) -> Dict:
    """
    Perform Mann-Whitney U test for independent samples.
    
    Args:
        data1: First sample
        data2: Second sample
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
    
    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
    
    # Calculate effect size (r)
    n1, n2 = len(data1), len(data2)
    z = stats.norm.ppf(p_value / 2) if alternative == "two-sided" else stats.norm.ppf(p_value)
    r = abs(z) / np.sqrt(n1 + n2)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_r": float(r),
        "alternative": alternative,
        "sample_size_1": n1,
        "sample_size_2": n2,
        "significant": p_value < 0.05
    }


def compare_variants(
    results_dict: Dict[str, List[Dict]],
    baseline_variant: str = "b3",
    metric: str = "PCS"
) -> Dict:
    """
    Compare multiple variants against a baseline using statistical tests.
    
    Args:
        results_dict: Dictionary mapping variant names to list of result dicts
        baseline_variant: Name of baseline variant
        metric: Metric to compare
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "baseline": baseline_variant,
        "metric": metric,
        "variants": {}
    }
    
    baseline_data = np.array([r[metric] for r in results_dict[baseline_variant]])
    
    for variant_name, results in results_dict.items():
        if variant_name == baseline_variant:
            continue
            
        variant_data = np.array([r[metric] for r in results])
        
        # Statistical test
        test_result = wilcoxon_signed_rank_test(variant_data, baseline_data)
        
        # Confidence intervals
        baseline_ci = calculate_confidence_interval(baseline_data)
        variant_ci = calculate_confidence_interval(variant_data)
        
        comparison["variants"][variant_name] = {
            "baseline_stats": baseline_ci,
            "variant_stats": variant_ci,
            "statistical_test": test_result,
            "improvement": float(np.mean(variant_data) - np.mean(baseline_data)),
            "improvement_percent": float(((np.mean(variant_data) - np.mean(baseline_data)) / np.mean(baseline_data)) * 100)
        }
    
    return comparison


def summarize_results(results: List[Dict], metrics: Optional[List[str]] = None) -> Dict:
    """
    Generate comprehensive statistical summary of results.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to summarize (defaults to common metrics)
    
    Returns:
        Dictionary with comprehensive summary
    """
    if metrics is None:
        metrics = ["PCS", "Fe1", "H", "delta_alpha", "latency_ms"]
    
    summary = {
        "sample_size": len(results),
        "metrics": {}
    }
    
    for metric in metrics:
        if metric not in results[0]:
            continue
            
        data = np.array([r[metric] for r in results])
        
        summary["metrics"][metric] = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "confidence_interval_95": calculate_confidence_interval(data, 0.95),
            "confidence_interval_99": calculate_confidence_interval(data, 0.99)
        }
    
    return summary


def compute_effect_size_cohen_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two independent samples.
    
    Args:
        data1: First sample
        data2: Second sample
    
    Returns:
        Cohen's d value
    """
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(data1) - np.mean(data2)) / pooled_std
    
    return float(d)


def bootstrap_confidence_interval(
    data: np.ndarray,
    num_iterations: int = 10000,
    confidence: float = 0.95,
    statistic: str = "mean"
) -> Dict:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Input data
        num_iterations: Number of bootstrap iterations
        confidence: Confidence level
        statistic: Statistic to compute ('mean', 'median')
    
    Returns:
        Dictionary with bootstrap results
    """
    bootstrapped_stats = []
    n = len(data)
    
    for _ in range(num_iterations):
        indices = np.random.randint(0, n, n)
        sample = data[indices]
        
        if statistic == "mean":
            bootstrapped_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrapped_stats.append(np.median(sample))
    
    bootstrapped_stats = np.array(bootstrapped_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_stats, (alpha / 2) * 100)
    upper = np.percentile(bootstrapped_stats, (1 - alpha / 2) * 100)
    
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "num_iterations": num_iterations,
        "statistic": statistic,
        "bootstrap_mean": float(np.mean(bootstrapped_stats)),
        "bootstrap_std": float(np.std(bootstrapped_stats))
    }
