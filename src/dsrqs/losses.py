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

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSRQSLoss(nn.Module):
    """
    Combined loss function for DSRQS:
        L = L_CE + λ * L_DC
    
    where:
        - L_CE: Binary cross-entropy loss on depth-conditional labels
        - L_DC: Depth-contrastive margin loss on conflicting predictions
        - λ: weight parameter (default: 0.4)
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.lam   = cfg["loss"]["lambda_dc"]      # λ = 0.4 (per paper)
        self.gamma = cfg["loss"]["margin"]         # margin γ = 0.25
        self.bce   = nn.BCELoss(reduction="mean")

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        hops:   torch.Tensor,
        qids:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            scores: Predicted scores from model (batch_size,)
            labels: Depth-conditional binary labels (batch_size,)
            hops: Hop depths for each sample (batch_size,)
            qids: Query IDs for grouping samples (batch_size,)
        
        Returns:
            loss: Combined L_CE + λ*L_DC
        """
        # ====================================================================
        # L_CE: Standard binary cross-entropy on depth-conditional labels
        # ====================================================================
        lce = self.bce(scores, labels.float())

        # ====================================================================
        # L_DC: Depth-contrastive margin loss
        # ====================================================================
        # Mine contrastive pairs: (Q, r, ℓ⁺, ℓ⁻) where:
        #   - Same query Q
        #   - Same relation type r
        #   - OPPOSITE labels at different depths: y_{ℓ⁺} ≠ y_{ℓ⁻}
        # 
        # Loss: max(0, g_{ℓ⁻} - g_{ℓ⁺} + γ)
        # This explicitly penalizes the model when it ranks the irrelevant
        # depth higher than the relevant depth.
        
        ldc = torch.tensor(0.0, device=scores.device)
        n_triplets = 0
        
        # Group by query
        for qid in torch.unique(qids):
            mask_qid = qids == qid
            
            # Get all samples for this query
            qid_scores = scores[mask_qid]
            qid_labels = labels[mask_qid]
            qid_hops   = hops[mask_qid]
            
            # Find positive (label=1) and negative (label=0) samples
            pos_mask = qid_labels == 1
            neg_mask = qid_labels == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Get scores for positive and negative samples
                pos_scores = qid_scores[pos_mask]
                neg_scores = qid_scores[neg_mask]
                
                # Compute contrastive loss: max(0, neg_score - pos_score + γ)
                # We want: pos_score - neg_score ≥ γ
                # Or equivalently: pos_score ≥ neg_score + γ
                
                # Expand dimensions for pairwise comparison
                # Shape: (n_pos, n_neg)
                pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
                neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)
                
                # Pairwise margin: max(0, neg - pos + γ)
                margins = torch.clamp(
                    neg_expanded - pos_expanded + self.gamma,
                    min=0.0
                )
                
                # Sum over all positive-negative pairs
                ldc += margins.sum()
                n_triplets += margins.numel()
        
        # Average over number of triplets (or batches if no pairs)
        if n_triplets > 0:
            ldc = ldc / n_triplets
        else:
            # If no contrastive pairs found, DC loss is zero
            ldc = torch.tensor(0.0, device=scores.device)
        
        # ====================================================================
        # Combined objective: L = L_CE + λ*L_DC
        # ====================================================================
        total_loss = lce + self.lam * ldc
        
        return total_loss