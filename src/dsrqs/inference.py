# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Algorithm 1: DSRQS Inference (Exact from paper)
#   Complexity: O(|R|(d² + Ldρ) + |E₀|d) with caching
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict
from src.dsrqs.model import DSRQS


def filter_subgraph(
    model: DSRQS,
    q_emb: torch.Tensor,
    edges_by_depth: Dict[int, List[Tuple]],
    relation_encoder,
    threshold: float = 0.5,
    cache: Optional[Dict] = None,
) -> Set[Tuple]:
    """
    Algorithm 1: DSRQS: Depth-Stratified Relation-Query Filtering
    
    Input:
        Q: Query
        {E_ℓ}_{ℓ=1 to L}: Edges by depth level
        parameters Φ: DSRQS model
        threshold θ: Filtering threshold (0.5 by default)
    
    Output:
        E_filt: Filtered edges
    
    Complexity:
        O(|R|(d² + Ldρ) + |E₀|d) with caching
    """
    device = q_emb.device

    if cache is None:
        cache = {}
    
    # Step 1: q ← Enc(Q)/∥Enc(Q)∥ (already normalized)
    q = q_emb.to(device)
    assert abs(q.norm().item() - 1.0) < 1e-6, "Query embedding should be L2 normalized!"

    # Step 2: E_filt ← ∅
    filtered_edges: Set[Tuple] = set()
    L = model.L

    # Step 3: For ℓ = 1 to L
    for l in range(1, L + 1):
        
        # Step 4: W_ℓ ← W₀ + A_ℓ B_ℓ^T (O(d²+dρ), once per depth)
        idx_0 = l - 1  # Convert 1-based to 0-based
        W_l = model.W0 + torch.matmul(model.A[idx_0], model.B[idx_0].T)

        if l not in edges_by_depth:
            continue

        # Step 5: For each (h, r, t) in E_ℓ
        for (h, r, t) in edges_by_depth[l]:
            
            # Step 6: If e_r not cached, e_r ← Enc(r)/∥Enc(r)∥, then cache
            if r not in cache:
                r_emb = relation_encoder(r)
                r_emb = r_emb / r_emb.norm(p=2)
                cache[r] = r_emb.to(device)
            
            er = cache[r]
            
            # Step 7: s ← q^T W_ℓ er + v^T(q ⊙ er) + b_ℓ
            base = torch.einsum("d, d, d", q, W_l, er)
            hadamard = torch.einsum("d, d, d", model.v, q, er)
            bias = model.b[idx_0]
            s = base + hadamard + bias
            
            # Step 8: If σ(s) ≥ θ, add to E_filt
            score = torch.sigmoid(s)
            if score.item() >= threshold:
                filtered_edges.add((h, r, t))
    
    # Step 9: Return E_filt
    return filtered_edges


def edges_by_depth_from_relations(
    full_graph: List[Tuple],
    query_depth_mapping: Optional[Dict] = None
) -> Dict[int, List[Tuple]]:
    """Group edges by their depth level as per the paper's dataset structure."""
    grouped = defaultdict(list)
    
    if query_depth_mapping is not None:
        # Use the provided query-specific depth mapping
        for (h, r, t), depth in query_depth_mapping.items():
            grouped[depth].append((h, r, t))
        return grouped
    
    # If no mapping, assume edge IDs contain depth information
    for edge in full_graph:
        h, r, t = edge[:3]
        depth = 1
        if len(edge) > 3 and isinstance(edge[3], int):
            depth = edge[3]
        grouped[depth].append(edge)
    
    return grouped
