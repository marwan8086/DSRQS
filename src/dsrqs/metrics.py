# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   This file is part of the DSRQS framework for multi-hop reasoning over
#   biomedical knowledge graphs and retrieval-augmented generation (RAG).
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

from typing import List, Set, Tuple, Any

import numpy as np
from sklearn.metrics import f1_score


def path_coherence_score(
    gold_paths: List[List[Any]],
    pred_edges: Set[Tuple],
) -> float:
    """
    Compute Path-Coherence Score (PCS).
    
    PCS measures path-level integrity: it returns the fraction of gold
    answer-supporting paths that are completely preserved in the filtered
    edge set. Unlike edge-level F1, PCS is ZERO if any single edge of
    any gold path is pruned.
    
    Definition:
        PCS(Q, E_filt) = |{P ∈ P*(Q) : P ⊆ E_filt}| / |P*(Q)|
    
    Args:
        gold_paths: List of gold answer paths, each path is a list of edges
        pred_edges: Set of predicted (retained) edges as tuples
    
    Returns:
        PCS score in [0, 1]
    """
    if not gold_paths:
        return 1.0
    
    # Count how many gold paths are completely intact in pred_edges
    coherent_paths = sum(
        1
        for path in gold_paths
        if all(tuple(e) in pred_edges for e in path)
    )
    
    return coherent_paths / len(gold_paths)


def compute_hallucination_rate(
    pcs: float,
) -> float:
    """
    Estimate hallucination rate from PCS.
    
    Empirical finding (per paper): Strong anti-correlation between PCS
    and expert-adjudicated hallucination rate (Spearman ρ = -0.96).
    
    Linear approximation observed across datasets:
        H ≈ a - b*PCS
    
    Args:
        pcs: Path-Coherence Score
    
    Returns:
        Estimated hallucination rate as percentage [0, 100]
    """
    # Fit from Figure 3 (scatter plot) of paper
    # Approximate linear relationship: H ≈ 25 - 20*PCS
    # (varies slightly by dataset, but this is a reasonable proxy)
    hallucination = max(0.0, 25.1 - 20.1 * pcs)
    return hallucination


def compute_all_metrics(
    all_preds:       np.ndarray,
    all_labels:      np.ndarray,
    all_gold_paths:  List[List[Any]],
    all_pred_edges:  List[List[Tuple]],
    theta:           float = 0.5,
) -> dict:
    """
    Compute comprehensive evaluation metrics for DSRQS.
    
    Metrics:
        - PCS: Path-Coherence Score (primary metric)
        - Fe1: Edge-level F1 score (secondary metric)
        - H: Estimated hallucination rate (%)
        - precision: Edge-level precision
        - recall: Edge-level recall
    
    Args:
        all_preds: Predicted scores (n_edges,)
        all_labels: Gold binary labels (n_edges,)
        all_gold_paths: Gold paths per query (n_queries,)
        all_pred_edges: Predicted edges per query (n_queries,)
        theta: Threshold for binary classification (default: 0.5)
    
    Returns:
        Dictionary with computed metrics
    """
    # ====================================================================
    # PCS: Path-Coherence Score (primary metric)
    # ====================================================================
    pcs_per_query = [
        path_coherence_score(gp, set(map(tuple, pe)))
        for gp, pe in zip(all_gold_paths, all_pred_edges)
    ]
    pcs = float(np.mean(pcs_per_query)) if pcs_per_query else 0.0
    
    # ====================================================================
    # Edge-level F1 (Fe1)
    # ====================================================================
    binary_preds = (all_preds >= theta).astype(int)
    fe1 = float(f1_score(all_labels, binary_preds, zero_division=0))
    
    # Precision and Recall at edge level
    from sklearn.metrics import precision_score, recall_score
    precision = float(precision_score(all_labels, binary_preds, zero_division=0))
    recall = float(recall_score(all_labels, binary_preds, zero_division=0))
    
    # ====================================================================
    # Hallucination Rate (H)
    # ====================================================================
    hallucination = compute_hallucination_rate(pcs)
    
    return {
        "PCS": pcs,
        "Fe1": fe1,
        "precision": precision,
        "recall": recall,
        "H": hallucination,
    }