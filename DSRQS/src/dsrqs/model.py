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

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        d   = cfg["model"]["hidden_dim"]
        rho = cfg["model"]["lora_rank"]
        L   = cfg["model"]["max_hops"]

        self.W0 = nn.Bilinear(d, d, 1, bias=False)
        self.A  = nn.ParameterList(
            [nn.Parameter(torch.empty(d, rho)) for _ in range(L + 1)]
        )
        self.B  = nn.ParameterList(
            [nn.Parameter(torch.empty(d, rho)) for _ in range(L + 1)]
        )
        self.v  = nn.Parameter(torch.empty(d))
        self.b  = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(L + 1)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        L = self.cfg["model"]["max_hops"]
        d = self.cfg["model"]["hidden_dim"]
        self.W0.weight.data = torch.eye(d).unsqueeze(0)
        for l in range(L + 1):
            nn.init.kaiming_uniform_(self.A[l], a=5 ** 0.5)
            nn.init.zeros_(self.B[l])
            nn.init.zeros_(self.b[l])
        nn.init.normal_(self.v, mean=0.0, std=1.0 / d)

    def forward(
        self,
        q:   torch.Tensor,
        r:   torch.Tensor,
        hop: torch.Tensor,
    ) -> torch.Tensor:
        base     = self.W0(q, r).squeeze(-1)
        hadamard = (self.v * q * r).sum(dim=-1)
        delta    = torch.zeros_like(base)
        bias     = torch.zeros_like(base)
        for i in range(q.size(0)):
            l        = int(hop[i].item())
            Wl_corr  = self.A[l] @ self.B[l].t()
            delta[i] = q[i] @ Wl_corr @ r[i]
            bias[i]  = self.b[l]
        logit = base + delta + hadamard + bias
        return torch.sigmoid(logit)