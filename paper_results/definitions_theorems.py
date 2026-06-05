# =============================================================================
# Paper Definitions and Theorems
# Complete formalization from Section 3 and Section 6
# =============================================================================

DEFINITIONS = [
    {
        "number": "3.1",
        "name": "Depth-Conditional Relevance",
        "latex": r"""
Definition 3.1 (Depth-Conditional Relevance). For query $Q$ and hop depth $\ell$, let $\mathcal{R}^*_\ell(Q)$ denote the set of relation types appearing on at least one gold answer-supporting path at depth $\ell$. Define $y_\ell(Q, r) = \mathbb{1}[r \in \mathcal{R}^*_\ell(Q)]$.
        """,
        "text": "Definition 3.1 (Depth-Conditional Relevance): For query Q and hop depth ℓ, let R*_ℓ(Q) denote the set of relation types appearing on at least one gold answer-supporting path at depth ℓ. Define y_ℓ(Q, r) = 1[r ∈ R*_ℓ(Q)]."
    },
    {
        "number": "3.2",
        "name": "Position-Conflation Error",
        "latex": r"""
Definition 3.2 (Position-Conflation Error). A depth-agnostic filter $f(Q, (h, r, t))$ commits a PCE on $(Q, r, \ell_1, \ell_2)$ if $\ell_1 \neq \ell_2$, $y_{\ell_1}(Q, r) \neq y_{\ell_2}(Q, r)$, yet $f$ makes the same binary decision for $r$ at both depths.
        """,
        "text": "Definition 3.2 (Position-Conflation Error): A depth-agnostic filter f(Q, (h, r, t)) commits a PCE on (Q, r, ℓ1, ℓ2) if ℓ1 ≠ ℓ2, y_ℓ1(Q, r) ≠ y_ℓ2(Q, r), yet f makes the same binary decision for r at both depths."
    },
    {
        "number": "3.3",
        "name": "PCE Rate",
        "latex": r"""
Definition 3.3 (PCE Rate). For dataset $\mathcal{D}$:
$$
\mathrm{PCE}(\mathcal{D}) = \frac{1}{|\mathcal{D}|} 
\sum_{(Q, \mathcal{E}_0) \in \mathcal{D}}
\frac{\sum_{\ell, r} \mathbb{1}[y_\ell(Q, r) \neq y_1(Q, r)] \cdot \mathbb{1}[\exists (h, r, t) \in E_\ell]}
{\sum_\ell |\mathcal{R}_\ell(Q)|}
$$
        """,
        "text": "Definition 3.3 (PCE Rate): For dataset D, PCE(D) is the fraction of relations that are position-conflated across the dataset."
    },
    {
        "number": "4.1",
        "name": "KG Structure",
        "text": "Assumption 4.1 (KG Structure): G = (V, E, R) is a typed directed multigraph with |R| ≥ 2."
    },
    {
        "number": "4.2",
        "name": "Encoder Expressivity",
        "text": "Assumption 4.2 (Encoder Expressivity): Enc : Q ∪ R → R^d satisfies E[q^⊤ e_r+] > E[q^⊤ e_r−] for r+ ∈ R*_ℓ(Q), r− ∉ R*_ℓ(Q)."
    },
    {
        "number": "4.3",
        "name": "Depth Identifiability",
        "text": "Assumption 4.3 (Depth Identifiability): The hop depth ℓ of every (h, r, t) ∈ E0 is computable in O(|E0|) by BFS from S(Q)."
    },
    {
        "number": "4.4",
        "name": "Path Decomposability",
        "text": "Assumption 4.4 (Path Decomposability): Every gold path P* = (e1, …, eL) satisfies eℓ ∈ Eℓ."
    },
    {
        "number": "4.5",
        "name": "Path-Coherence Score",
        "latex": r"""
Definition 4.5 (Path-Coherence Score).
$$
\mathrm{PCS}(Q, E_{\text{filt}}) = 
\frac{|\{P \in P^*(Q) : P \subseteq E_{\text{filt}}\}|}{|P^*(Q)|},
$$
$$
\mathrm{PCS} = \frac{1}{|\mathcal{D}|} 
\sum_{(Q, E_{\text{filt}}) \in \mathcal{D}} \mathrm{PCS}(Q, E_{\text{filt}}).
$$
        """,
        "text": "Definition 4.5 (Path-Coherence Score): PCS measures the fraction of complete gold answer paths preserved. Unlike edge-level F1, PCS is zero if any single edge of any gold path is pruned."
    },
    {
        "number": "6.1",
        "name": "Path-Coherence Lower Bound",
        "type": "Theorem",
        "text": "Theorem 6.1 (Path-Coherence Lower Bound): Let α_ℓ = P[(h, r, t) ∈ E_filt | (h, r, t) ∈ E_ℓ, r ∈ R*_ℓ(Q)]. Under Assumption 4.4 and conditional independence of filtering across depths: E[PCS(Q, E_filt)] ≥ product_{ℓ=1}^L α_ℓ."
    },
    {
        "number": "6.2",
        "name": "Structural Inferiority",
        "type": "Theorem",
        "text": "Theorem 6.2 (Structural Inferiority): Let α*_ℓ (resp. ~α_ℓ) be the per-depth TPR of the optimal depth-aware (resp. any depth-agnostic) filter. If I(Y; L | Q, R) > 0, then sum_{ℓ} ~α_ℓ < sum_{ℓ} α*_ℓ and ∃ ℓ† with ~α_{ℓ†} < α*_{ℓ†}."
    },
    {
        "number": "6.3",
        "name": "Multiplicative PCS Gap",
        "type": "Corollary",
        "text": "Corollary 6.3 (Multiplicative PCS Gap): E[PCS_agnostic] ≤ product_{ℓ} ~α_ℓ < product_{ℓ} α*_ℓ ≤ E[PCS_DSRQS]."
    },
    {
        "number": "6.4",
        "name": "DC-Loss Drives Depth Separation",
        "type": "Proposition",
        "text": "Proposition 6.4 (DC-Loss Drives Depth Separation): Let Δ = g(Q, r, ℓ+) - g(Q, r, ℓ-) with Δ < γ. Under gradient descent (step η) on L_DC, Δ increases by 2η per step until Δ ≥ γ."
    }
]

MUTUAL_INFORMATION_ESTIMATE = {
    "point": 0.173,
    "std": 0.041,
    "ci_95_lower": 0.093,
    "ci_95_upper": 0.251
}

DATASETS = [
    {
        "name": "Orphanet-FQ274",
        "release": "Orphanet release 4.1 (January 2026)",
        "description": "10,493 rare diseases, 4,381 gene-disease associations, 8 relation types",
        "license": "CC-BY-4.0",
        "kappa": 0.81
    },
    {
        "name": "DisGeNET-RD411",
        "release": "DisGeNET v7.0",
        "description": "1,134,942 gene-disease associations for 21,671 diseases, 12 relation types",
        "license": "CC-BY-NC-SA-4.0",
        "kappa": 0.79
    },
    {
        "name": "OMIM-Hop3",
        "release": "OMIM",
        "description": "183 curated three-hop rare-disease QA instances, 9 relation types",
        "license": "Academic Use Only",
        "kappa": 0.82
    }
]


def print_all_definitions():
    """Print all formal definitions and theorems from paper."""
    print("=" * 100)
    print("DSRQS: Formal Definitions and Theorems")
    print("=" * 100)
    print()

    for defn in DEFINITIONS:
        print(f"  {defn.get('type', 'Definition')} {defn['number']}: {defn['name']}")
        print("-" * 100)
        print(defn['text'])
        print()
    print("=" * 100)
    print("\nMutual Information Estimate:")
    print(f"  I(Y; L | Q, R) = {MUTUAL_INFORMATION_ESTIMATE['point']} ± {MUTUAL_INFORMATION_ESTIMATE['std']} bits")
    print(f"  95% CI: [{MUTUAL_INFORMATION_ESTIMATE['ci_95_lower']}, {MUTUAL_INFORMATION_ESTIMATE['ci_95_upper']}]")
    print()

    print("Datasets:")
    print("-" * 100)
    for ds in DATASETS:
        print(f"  {ds['name']}")
        print(f"    Description: {ds['description']}")
        print(f"    License: {ds['license']}")
        print(f"    IRR (Cohen's κ): {ds['kappa']}")
        print()


if __name__ == "__main__":
    print_all_definitions()
