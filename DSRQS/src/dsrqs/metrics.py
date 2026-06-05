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
    if not gold_paths:
        return 1.0
    coherent = sum(
        1
        for path in gold_paths
        if all(tuple(e) in pred_edges for e in path)
    )
    return coherent / len(gold_paths)


def compute_all_metrics(
    all_preds:       np.ndarray,
    all_labels:      np.ndarray,
    all_gold_paths:  List[List[Any]],
    all_pred_edges:  List[List[Tuple]],
    theta:           float,
) -> dict:
    pcs_per_query = [
        path_coherence_score(gp, set(map(tuple, pe)))
        for gp, pe in zip(all_gold_paths, all_pred_edges)
    ]
    pcs = float(np.mean(pcs_per_query)) if pcs_per_query else 0.0

    binary_preds = (all_preds >= theta).astype(int)
    fe1 = float(f1_score(all_labels, binary_preds, zero_division=0))

    h = 25.1 - 20.1 * pcs

    return {"PCS": pcs, "Fe1": fe1, "H": h}