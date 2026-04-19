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

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.lam   = cfg["loss"]["lambda_dc"]
        self.gamma = cfg["loss"]["margin"]
        self.bce   = nn.BCELoss(reduction="mean")

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        qids:   torch.Tensor,
    ) -> torch.Tensor:

        lce = self.bce(scores, labels.float())

        ldc      = torch.tensor(0.0, device=scores.device)
        n_groups = 0

        for qid in torch.unique(qids):
            mask       = qids == qid
            s_group    = scores[mask]
            y_group    = labels[mask]
            pos_scores = s_group[y_group == 1]
            neg_scores = s_group[y_group == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                delta    = pos_scores.mean() - neg_scores.mean()
                ldc     += F.relu(self.gamma - delta)
                n_groups += 1

        if n_groups > 0:
            ldc = ldc / n_groups

        return lce + self.lam * ldc