import torch
from src.dsrqs.losses import DSRQSLoss


def test_dc_requires_opposite_labels():
    cfg = {"loss": {"lambda_dc": 0.4, "margin": 0.25}}
    loss_fn = DSRQSLoss(cfg)
    out = loss_fn(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1, 1]),
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
    )
    assert out["l_dc"].item() == 0.0
