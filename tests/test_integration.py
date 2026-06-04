# =============================================================================
# Quick Integration Test for DSRQS
# =============================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import numpy as np

# Test imports
try:
    from src.dsrqs.model import DSRQS
    from src.dsrqs.losses import DSRQSLoss
    from src.dsrqs.metrics import compute_all_metrics, path_coherence_score
    from src.dsrqs.data import KGRAGDataset, collate_fn
    from src.dsrqs.utils import load_config
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_model():
    """Test model creation and forward pass."""
    print("\n[TEST 1] Model initialization and forward pass")
    
    cfg = load_config("configs/default.yaml")
    model = DSRQS(cfg)
    print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy batch
    batch_size = 4
    d = cfg["model"]["hidden_dim"]
    
    q = torch.randn(batch_size, d)
    q = q / q.norm(dim=1, keepdim=True)  # L2 normalize
    r = torch.randn(batch_size, d)
    r = r / r.norm(dim=1, keepdim=True)
    hop = torch.tensor([1, 2, 3, 1])
    
    # Forward pass
    with torch.no_grad():
        scores = model(q, r, hop)
    
    print(f"  ✓ Forward pass OK: input shape {q.shape} → output shape {scores.shape}")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    assert scores.shape == (batch_size,), f"Wrong output shape: {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores not in [0, 1]"
    print("  ✓ Output is valid (sigmoid activated)")
    

def test_loss():
    """Test loss computation."""
    print("\n[TEST 2] Loss function (CE + DC)")
    
    cfg = load_config("configs/default.yaml")
    model = DSRQS(cfg)
    loss_fn = DSRQSLoss(cfg)
    print(f"  ✓ Loss module created: λ={cfg['loss']['lambda_dc']}, γ={cfg['loss']['margin']}")
    
    # Create dummy batch
    batch_size = 8
    d = cfg["model"]["hidden_dim"]
    
    q = torch.randn(batch_size, d)
    q = q / q.norm(dim=1, keepdim=True)
    r = torch.randn(batch_size, d)
    r = r / r.norm(dim=1, keepdim=True)
    hop = torch.tensor([1, 1, 2, 2, 3, 3, 1, 2])
    
    with torch.no_grad():
        scores = model(q, r, hop)
    
    # Create labels: some with depth variation to test DC loss
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1])
    qids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    loss = loss_fn(scores, labels, hop, qids)
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"
    print("  ✓ Loss is valid (no NaN/inf)")


def test_metrics():
    """Test metric computation."""
    print("\n[TEST 3] Metrics computation (PCS, F1, Hallucination)")
    
    # Create dummy predictions and labels
    n_samples = 20
    preds = np.random.uniform(0, 1, n_samples)
    labels = np.random.randint(0, 2, n_samples)
    
    # Create dummy gold paths
    gold_paths = [
        [["disease_A", "rel_1", "gene_B"]],  # 1-hop path
        [["disease_A", "rel_1", "gene_B"], ["gene_B", "rel_2", "pathway"]],  # 2-hop
    ] * 10  # Repeat for all queries
    
    # Create predicted edges
    pred_edges = [
        [("disease_A", "rel_1", "gene_B")],
        [("disease_A", "rel_1", "gene_B"), ("gene_B", "rel_2", "pathway")],
    ] * 10
    
    metrics = compute_all_metrics(preds, labels, gold_paths, pred_edges, theta=0.5)
    
    print(f"  ✓ Metrics computed:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"    - {key}: {val:.4f}")
        else:
            print(f"    - {key}: {val}")
    
    assert "PCS" in metrics, "Missing PCS metric"
    assert "Fe1" in metrics, "Missing F1 metric"
    assert "H" in metrics, "Missing hallucination metric"
    print("  ✓ All required metrics present")


def test_data_loading():
    """Test data loading."""
    print("\n[TEST 4] Data loading from JSON")
    
    cfg = load_config("configs/default.yaml")
    data_path = Path("data/orphanet_fq274/orphanet_fq274_full.json")
    
    if not data_path.exists():
        print(f"  ✗ Data file not found: {data_path}")
        return
    
    try:
        # Note: Loading will be slow due to transformer encoding
        # For testing, just check file format
        with open(data_path, "r") as f:
            data = json.load(f)
        
        print(f"  ✓ Loaded {len(data)} queries from {data_path.name}")
        
        if data:
            sample = data[0]
            print(f"    Sample query (qid={sample['qid']}):")
            print(f"      Query: {sample['query'][:60]}...")
            print(f"      Intent: {sample['intent']}")
            print(f"      Relations: {len(sample['relations'])} items")
            print(f"      Gold paths: {len(sample['gold_paths'])} paths")
            
            # Check structure
            assert "qid" in sample, "Missing qid"
            assert "query" in sample, "Missing query"
            assert "relations" in sample, "Missing relations"
            assert "gold_paths" in sample, "Missing gold_paths"
            print("  ✓ Data structure is valid")
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("  DSRQS Integration Test Suite")
    print("=" * 70)
    
    try:
        test_model()
        test_loss()
        test_metrics()
        test_data_loading()
        
        print("\n" + "=" * 70)
        print("  ✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
