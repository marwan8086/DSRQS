# DSRQS Quick Start Guide

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic benchmark data
python scripts/generate_benchmark_data.py

# 3. Validate data format
python scripts/validate_and_fix_data.py
```

## Quick Test

```bash
# Run integration tests
pytest tests/test_integration.py -v

# Should see 4 tests pass:
# ✓ test_model: Model creation and forward pass
# ✓ test_loss: Combined loss computation
# ✓ test_metrics: PCS and hallucination metrics
# ✓ test_data_loading: JSON data format
```

## Training

### Single Fold (Quick Test)
```bash
python main.py --dataset orphanet_fq274 --mode single_run --fold 0 --seed 0
```

Expected output:
```
Result → PCS=0.xxx  Fe1=0.xxx  H=xx.x%
```

### Full Evaluation (5-Fold CV × 5 Seeds = 25 models)
```bash
python main.py --dataset orphanet_fq274 --mode full_eval
```

This runs:
- 5 random seeds
- 5-fold cross-validation per seed
- ~25 models total
- Reports: PCS (primary), Edge F1, Hallucination rate

### Available Datasets
- `orphanet_fq274`: 50 queries (rare diseases)
- `disgenet_rd411`: 50 queries (gene-disease)
- `omim_hop3`: 30 queries (3-hop reasoning)

## Key Metrics

### Path-Coherence Score (PCS)
- **Definition**: Fraction of complete gold paths preserved in filtered edges
- **Range**: [0, 1] (1 = perfect)
- **Property**: Zero if ANY edge of ANY path is missing (chain integrity)
- **Expected**: 0.70-0.78 on benchmarks

### Hallucination Rate (H)
- **Definition**: Estimated % of LLM generation errors
- **Calibration**: H ≈ 25 - 20×PCS (from paper Figure 3)
- **Expected**: 7-11% with DSRQS (vs 10-24% with baselines)

### Edge F1
- **Definition**: Standard F1 on individual edge classification
- **Threshold**: θ=0.5
- **Expected**: 0.79-0.84

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  hidden_dim: 768         # Embedding dimension
  lora_rank: 16           # LoRA rank ρ
  max_hops: 3             # Maximum hop depth

loss:
  lambda_dc: 0.4          # DC loss weight
  margin: 0.25            # Contrastive margin γ

train:
  lr: 5.0e-4              # Learning rate
  batch_size: 32          # Batch size
  epochs: 10              # Training epochs

eval:
  threshold: 0.5          # Classification threshold θ
```

## Architecture

```
Input: Query Q, Relation r, Depth ℓ
         ↓
Encoder: BioLinkBERT-Large (frozen)
         ↓
L2 Normalize: q, e_r ← R^d
         ↓
DSRQS Scoring:
  • Bilinear: q^T @ (W_0 + A_ℓB_ℓ^T) @ e_r
  • Hadamard:  v^T ⊙ (q ⊙ e_r)
  • Bias:      b_ℓ
         ↓
Sigmoid: σ(·) → [0, 1]
         ↓
Output: g(Q, (h,r,t), ℓ) ∈ [0, 1]
```

## Loss Function

```
L = L_CE + λ·L_DC

L_CE:  Standard binary cross-entropy on depth-conditional labels
L_DC:  Margin loss on (Q,r,ℓ⁺,ℓ⁻) triplets with opposite labels
       max(0, g_{ℓ⁻} - g_{ℓ⁺} + γ)

λ=0.4, γ=0.25 (per paper)
```

## Inference

For a new query:

```python
from src.dsrqs.model import DSRQS
from src.dsrqs.utils import load_config
import torch

cfg = load_config("configs/default.yaml")
model = DSRQS(cfg)
model.load_state_dict(torch.load("checkpoints/seed0/fold0/best.pt"))
model.eval()

# For each relation at each depth
q_emb = encode_query(query)  # L2 normalized
e_r = encode_relation(relation)  # L2 normalized

with torch.no_grad():
    score = model(q_emb, e_r, torch.tensor([depth]))
    # score ∈ [0, 1]
    
    if score >= 0.5:  # threshold θ=0.5
        keep_edge = True
```

## Troubleshooting

### Q: GPU memory error
**A**: Reduce `batch_size` in `configs/default.yaml` (default: 32)

### Q: Slow data loading
**A**: First time loads embeddings from BioLinkBERT. Cache is built automatically.

### Q: Low PCS scores
**A**: 
- Check data has depth-conditional labels
- Verify gold_paths are correctly formatted
- Increase epochs for more training

### Q: Training diverges
**A**:
- Reduce learning rate: `lr: 1.0e-4`
- Reduce DC loss weight: `lambda_dc: 0.2`
- Check normalization: all embeddings should be L2 normalized

## Monitoring Training

Training logs are saved to `runs/run_YYYYMMDD_HHMMSS.json`

Key metrics tracked:
- Loss (CE and DC components separately)
- Validation PCS
- Validation F1
- Hallucination rate

## Paper Reproduction

To replicate paper results exactly:

1. Use real Orphanet-FQ274, DisGeNET-RD411, OMIM-Hop3 datasets
2. Run with seeds [0, 1, 2, 3, 4]
3. 5-fold CV per seed
4. Report: mean ± std, Wilcoxon tests ($p<0.05$)

Expected performance:
- **Orphanet**: PCS 0.738 ± 0.021, H 7.8%
- **DisGeNET**: PCS 0.768 ± 0.018, H 8.4%  
- **OMIM-Hop3**: PCS 0.714 ± 0.026, H 10.9%

---

For detailed documentation, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
