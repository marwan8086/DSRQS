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

import torch
import torch.nn as nn
from typing import Dict


class DSRQS(nn.Module):
    """
    Depth-Stratified Relation-Query Scoring (DSRQS).
    
    Implements the scoring function:
        g(Q, (h,r,t), ℓ) = σ(q^T(W_0 + A_ℓB_ℓ^T)e_r + v^T(q⊙e_r) + b_ℓ)
    
    where:
        - W_0: shared base interaction matrix (d × d, initialized as I)
        - A_ℓ, B_ℓ: depth-specific low-rank factors (d × ρ each)
        - v: shared Hadamard weight vector (d)
        - b_ℓ: depth-specific bias scalar
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        d   = cfg["model"]["hidden_dim"]
        rho = cfg["model"]["lora_rank"]
        L   = cfg["model"]["max_hops"]

        # Shared base interaction matrix (d × d)
        self.W0 = nn.Parameter(torch.empty(d, d))
        
        # Depth-specific LoRA factors: A_ℓ, B_ℓ ∈ ℝ^(d × ρ)
        self.A  = nn.ParameterList(
            [nn.Parameter(torch.empty(d, rho)) for _ in range(L + 1)]
        )
        self.B  = nn.ParameterList(
            [nn.Parameter(torch.empty(d, rho)) for _ in range(L + 1)]
        )
        
        # Shared Hadamard weight (d)
        self.v  = nn.Parameter(torch.empty(d))
        
        # Depth-specific biases (scalar per depth)
        self.b  = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(L + 1)]
        )
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize parameters according to paper specifications:
        - W_0 = I (identity matrix)
        - A_ℓ ~ N(0, 1/d)
        - B_ℓ = 0 (initially)
        - v ~ N(0, 1/d)
        - b_ℓ = 0
        """
        L = self.cfg["model"]["max_hops"]
        d = self.cfg["model"]["hidden_dim"]
        
        # W_0 = Identity matrix
        nn.init.eye_(self.W0)
        
        # LoRA factors and biases per depth
        for l in range(L + 1):
            # A_ℓ ~ N(0, 1/d)
            nn.init.normal_(self.A[l], mean=0.0, std=1.0 / d)
            # B_ℓ = 0 initially (so W_ℓ = W_0 at start)
            nn.init.zeros_(self.B[l])
            # b_ℓ = 0
            nn.init.zeros_(self.b[l])
        
        # v ~ N(0, 1/d)
        nn.init.normal_(self.v, mean=0.0, std=1.0 / d)

    def forward(
        self,
        q:   torch.Tensor,
        r:   torch.Tensor,
        hop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation relevance scores.
        
        Args:
            q: Query embeddings (batch_size × d), L2-normalized
            r: Relation embeddings (batch_size × d), L2-normalized
            hop: Hop depths (batch_size,)
        
        Returns:
            scores: Sigmoid-activated scores (batch_size,)
        """
        batch_size = q.size(0)
        device = q.device
        
        # Bilinear term: q^T @ W_0 @ e_r
        # Shape: (batch_size,)
        base = torch.einsum('bi,ij,bj->b', q, self.W0, r)
        
        # Depth-stratified LoRA term: q^T @ (A_ℓ @ B_ℓ^T) @ e_r
        # for each sample i: q_i^T @ (A_{hop_i} @ B_{hop_i}^T) @ r_i
        delta = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            hop_i = int(hop[i].item())
            # Compute W_ℓ correction: A_ℓ @ B_ℓ^T
            W_corr = self.A[hop_i] @ self.B[hop_i].t()  # (d × d)
            # q_i^T @ W_corr @ r_i
            delta[i] = torch.einsum('d,de,e->', q[i], W_corr, r[i])
        
        # Hadamard term: v^T @ (q ⊙ r)
        # Shape: (batch_size,)
        hadamard = torch.einsum('d,bd,bd->b', self.v, q, r)
        
        # Depth-specific bias: b_ℓ
        bias = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            hop_i = int(hop[i].item())
            bias[i] = self.b[hop_i].squeeze()
        
        # Combine all components: base + delta + hadamard + bias
        logits = base + delta + hadamard + bias
        
        # Apply sigmoid activation
        return torch.sigmoid(logits)