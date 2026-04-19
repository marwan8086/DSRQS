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
import torch
import pytest
import numpy as np
from src.dsrqs.model   import DSRQS
from src.dsrqs.losses  import DSRQSLoss
from src.dsrqs.metrics import path_coherence_score, compute_all_metrics


@pytest.fixture
def cfg():
    return {
        "model": {"hidden_dim": 64, "lora_rank": 4, "max_hops": 3},
        "loss":  {"lambda_dc": 0.4, "margin": 0.25},
    }


class TestDSRQSModel:

    def test_output_shape(self, cfg):
        model = DSRQS(cfg)
        out   = model(torch.randn(8, 64), torch.randn(8, 64),
                      torch.randint(1, 4, (8,)))
        assert out.shape == (8,)

    def test_output_range(self, cfg):
        model = DSRQS(cfg)
        out   = model(torch.randn(8, 64), torch.randn(8, 64),
                      torch.randint(1, 4, (8,)))
        assert torch.all(out >= 0) and torch.all(out <= 1)

    def test_no_nan(self, cfg):
        model = DSRQS(cfg)
        out   = model(torch.randn(4, 64), torch.randn(4, 64),
                      torch.tensor([1, 2, 3, 1]))
        assert not torch.any(torch.isnan(out))

    def test_grad_flows(self, cfg):
        model = DSRQS(cfg)
        q, r  = torch.randn(4, 64), torch.randn(4, 64)
        hop   = torch.tensor([1, 2, 3, 1])
        model(q, r, hop).sum().backward()
        assert model.W0.weight.grad is not None
        assert model.v.grad         is not None


class TestDSRQSLoss:

    def test_basic(self, cfg):
        loss_fn = DSRQSLoss(cfg)
        loss = loss_fn(torch.tensor([0.8, 0.2]),
                       torch.tensor([1,   0]),
                       torch.tensor([0,   0]))
        assert not torch.isnan(loss)

    def test_mask_bug1_fixed(self, cfg):
        loss_fn = DSRQSLoss(cfg)
        loss = loss_fn(torch.tensor([0.8, 0.1, 0.9, 0.2]),
                       torch.tensor([1,   0,   1,   0]),
                       torch.tensor([0,   0,   1,   1]))
        assert not torch.isnan(loss) and loss.item() >= 0


class TestMetrics:

    def test_pcs_full(self):
        gp = [[(0, "r", 1), (0, "s", 2)]]
        pe = {(0, "r", 1), (0, "s", 2)}
        assert path_coherence_score(gp, pe) == 1.0

    def test_pcs_miss(self):
        gp = [[(0, "r", 1), (0, "s", 2)]]
        pe = {(0, "r", 1)}
        assert path_coherence_score(gp, pe) == 0.0

    def test_pcs_empty(self):
        assert path_coherence_score([], set()) == 1.0

    def test_metrics_keys(self):
        m = compute_all_metrics(np.array([0.8, 0.2]),
                                np.array([1,   0]),
                                [[], []], [[], []], 0.5)
        assert set(m.keys()) == {"PCS", "Fe1", "H"}