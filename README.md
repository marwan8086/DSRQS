# Depth-Stratified Relation-Query Scoring (DSRQS)

## Overview

DSRQS is a depth-aware relation filtering framework designed to eliminate the Position-Conflation Error (PCE) in multi-hop Knowledge Graph Retrieval-Augmented Generation (KG-RAG) systems for biomedical applications.

The core issue addressed is that relation relevance is not invariant across graph depth, yet most existing models assume depth-invariant scoring, leading to structural reasoning errors in multi-hop biomedical inference.

---

## Problem Definition

We define the Position-Conflation Error (PCE) as a structural failure mode where a relation is assigned identical relevance scores across different hop depths despite having different semantic roles.

Empirical findings:
- 49.3% of relation types change relevance across hops
- 35–41% of filtering errors in state-of-the-art systems are due to PCE
- Depth-agnostic models suffer a provable information-theoretic loss when I(Y; L | Q, R) > 0

---

## Contributions

- Formal definition and theoretical grounding of Position-Conflation Error
- Depth-stratified relation scoring via LoRA decomposition
- Depth-contrastive learning objective for hop-aware separation
- Path-Coherence Score (PCS) as a structural evaluation metric
- OMIM-Hop3 benchmark for 3-hop rare disease reasoning
- Theoretical guarantees on PCS lower bound and structural inferiority of depth-agnostic models

---

## Method

For query Q and relation r at depth ℓ:

g(Q, r, ℓ) = σ(
qᵀ (W₀ + A_ℓ B_ℓᵀ) e_r
+ vᵀ (q ⊙ e_r)
+ b_ℓ
)

Where:
- W₀: shared base interaction matrix
- A_ℓ, B_ℓ: low-rank depth-specific adaptation (LoRA)
- v: Hadamard interaction vector
- b_ℓ: depth bias
- q, e_r: normalized embeddings

---

## Architecture

- Frozen biomedical encoder (BioLinkBERT-Large)
- Depth router selecting (A_ℓ, B_ℓ, b_ℓ)
- LoRA-based bilinear interaction per depth
- Hadamard interaction component
- Sigmoid scoring with threshold filtering

---

## Datasets

Orphanet:
- Rare disease ontology
- CC BY 4.0

DisGeNET:
- Gene-disease associations
- CC BY-NC-SA 4.0

OMIM:
- Mendelian disease knowledge graph
- Academic licensed dataset

OMIM-Hop3 Benchmark:
- 183 curated 3-hop reasoning instances
- Expert-validated gold paths

---

## Training Setup

- Encoder: BioLinkBERT-Large (frozen)
- LoRA rank: 16
- Optimizer: Adam
- Learning rate: 5e-4
- Loss: Cross-Entropy + Depth-Contrastive Loss
- Margin: 0.25
- Threshold: 0.5
- Evaluation: 5-fold cross-validation × 5 seeds
- Hardware: NVIDIA A100 80GB

---

## Results

DSRQS achieves:

- Higher Path-Coherence Score (PCS)
- Significant reduction in hallucination rate
- Strong improvements over cosine, bilinear, and contrastive baselines
- Stable performance across Orphanet, DisGeNET, and OMIM-Hop3

Key observation:
PCS is strongly anti-correlated with hallucination (ρ ≈ −0.96)

---

## Ablation

- Removing LoRA reduces performance significantly
- Removing depth-contrastive loss yields equivalent degradation
- Depth stratification is the dominant factor in performance gain
- Model is robust to moderate LoRA rank variation

---

## Theoretical Insights

- Depth-agnostic filtering is suboptimal under I(Y; L | Q, R) > 0
- PCE introduces multiplicative degradation across hops
- PCS lower bound scales as product of per-depth recall probabilities
- Structural inferiority theorem proves strict dominance of depth-aware filtering

---

## Limitations

- Fixed maximum reasoning depth (L = 3)
- Uniform depth treatment across heterogeneous graph topology
- Annotation cost for gold-path construction
- Evaluation limited to single biomedical KG setting

---

## Reproducibility

- Python 3.10
- PyTorch 2.1
- CUDA 12.1
- Transformers 4.36

All experiments are reproducible with fixed seeds and documented hyperparameters.

---

## Citation

Dhifallah & Liu. Depth-Stratified Relation-Query Scoring (DSRQS): Eliminating Position-Conflation Error in Biomedical KG-RAG Systems.

---

## License

Code: MIT License  
Data: Subject to Orphanet, DisGeNET, and OMIM licensing terms
