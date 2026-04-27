# Depth-Stratified Relation-Query Scoring (DSRQS)

## Overview

DSRQS is a depth-stratified relation filtering framework designed to eliminate the **Position-Conflation Error (PCE)** in multi-hop biomedical knowledge graph retrieval.

The method targets a fundamental limitation in existing relation filtering approaches:  
**relation relevance is not invariant across hop depth**, yet most models assume depth-invariant scoring.  
This mismatch leads to structural reasoning errors in multi-hop inference.

Importantly, DSRQS operates strictly at the **retrieval (relation-filtering) level** within KG-RAG pipelines and does not include answer generation.

---

## Problem Definition

The **Position-Conflation Error (PCE)** arises when a relation type is assigned identical relevance across different hop depths despite having distinct semantic roles.

Empirical observations:

- 49.3% of relation types change relevance across hops  
- 35–41% of filtering errors in state-of-the-art systems are attributable to PCE  
- Depth-agnostic models incur provable information loss when  
  I(Y; L | Q, R) > 0  

---

## Contributions

- Formal definition and theoretical characterization of PCE  
- Depth-stratified relation scoring via LoRA-based decomposition  
- Depth-contrastive learning objective for hop-aware discrimination  
- Path-Coherence Score (PCS) for evaluating structural reasoning integrity  
- OMIM-Hop3 benchmark for 3-hop rare disease reasoning  
- Theoretical guarantees on PCS lower bounds and structural inferiority of depth-agnostic filters  

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
- b_ℓ: depth-specific bias  
- q, e_r: ℓ2-normalized embeddings  

---

## Architecture

- Frozen biomedical encoder (**BioLinkBERT-Large**)  
- Depth router selecting (A_ℓ, B_ℓ, b_ℓ)  
- Depth-stratified bilinear interaction (LoRA-based)  
- Hadamard interaction component  
- Sigmoid scoring with threshold-based filtering  

---

## Datasets

- **Orphanet**  
  Rare disease knowledge base (CC BY 4.0)

- **DisGeNET**  
  Gene–disease association graph (CC BY-NC-SA 4.0)

- **OMIM**  
  Mendelian disease knowledge base (academic license required)

- **OMIM-Hop3 Benchmark**  
  - 183 curated 3-hop reasoning instances  
  - Expert-validated gold paths  

---

## Training Setup

- Encoder: BioLinkBERT-Large (frozen)  
- LoRA rank: 16  
- Optimizer: Adam  
- Learning rate: 5 × 10⁻⁴  
- Loss: Cross-Entropy + Depth-Contrastive Loss  
- Margin γ: 0.25  
- Threshold θ: 0.5  
- Evaluation: 5-fold cross-validation × 5 seeds  
- Hardware: NVIDIA A100 (80GB)  

---

## Results

DSRQS demonstrates:

- Significant improvements in **Path-Coherence Score (PCS)**  
- Reduction in hallucination rate  
- Consistent gains over cosine, bilinear, and contrastive baselines  
- Stable performance across Orphanet, DisGeNET, and OMIM-Hop3  

Key finding:

- PCS exhibits strong negative correlation with hallucination  
  (Spearman ρ ≈ −0.96)

---

## Ablation Study

- Removing LoRA reduces performance significantly  
- Removing depth-contrastive loss results in comparable degradation  
- Depth stratification is the dominant performance factor  
- Model remains robust across moderate LoRA rank variations  

---

## Theoretical Insights

- Depth-agnostic filtering is suboptimal when I(Y; L | Q, R) > 0  
- PCE induces multiplicative degradation across hops  
- PCS lower bound scales as the product of per-depth recall  
- Structural Inferiority Theorem proves dominance of depth-aware filtering  

---

## Limitations

- Fixed maximum reasoning depth (L = 3)  
- Uniform treatment across heterogeneous graph topology  
- High annotation cost for gold-path construction  
- Evaluation limited to single-KG biomedical setting  

---

## Reproducibility

- Python 3.10  
- PyTorch 2.1  
- CUDA 12.1  
- Transformers 4.36  

All experiments are fully reproducible with fixed seeds and documented hyperparameters.

---

## Development Environments

The project supports two official execution environments:

- **Google Colab**  
  For cloud-based execution and rapid experimentation  

- **Visual Studio Code (Local Setup)**  
  For full pipeline execution and reproducible experimentation  

Note:  
Both setups require access to a computational backend (GPU-enabled environment or server), due to the computational demands of multi-hop knowledge graph processing.

Both implementations are fully consistent and yield identical results under the same configuration.

---

## Scope Clarification

DSRQS is a **retrieval-layer method** and does not perform answer generation.

The framework focuses exclusively on **depth-aware relation filtering** and evaluates structural reasoning quality using PCS and edge-level metrics.  
It is intended to be integrated as a component within larger KG-RAG systems.

---

## Citation

Dhifallah, M., & Liu.  
*Depth-Stratified Relation-Query Scoring (DSRQS): Eliminating Position-Conflation Error in Multi-Hop Biomedical KG Retrieval.*

---

## License

- Code: MIT License  
- Data: Subject to Orphanet, DisGeNET, and OMIM licensing terms  
