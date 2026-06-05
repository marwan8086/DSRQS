# =============================================================================
# Paper Definitions and Theorems
# =============================================================================
DEFINITIONS = [
    {
        "type": "Definition",
        "number": "3.1",
        "name": "Depth-Conditional Relevance",
        "content": "For query Q and hop depth l, let R*_l(Q) denote the set of relation types appearing on at least one gold answer-supporting path at depth l. Define y_l(Q, r) = 1[r ∈ R*_l(Q)]."
    },
    {
        "type": "Definition",
        "number": "3.2",
        "name": "Position-Conflation Error",
        "content": "A depth-agnostic filter f(Q, (h, r, t)) commits a PCE on (Q, r, l1, l2) if l1 != l2, y_l1(Q, r) != y_l2(Q, r), yet f makes the same binary decision for r at both depths."
    },
    {
        "type": "Definition",
        "number": "3.3",
        "name": "PCE Rate",
        "content": "For dataset D: PCE(D) = (1/|D|) * sum_{(Q,E0)∈D} [ sum_{l,r} 1[y_l(Q,r)!=y_1(Q,r)] * 1[∃(h,r,t)∈El] / sum_{l} |R_l(Q)| ]"
    },
    {
        "type": "Assumption",
        "number": "4.1",
        "name": "KG Structure",
        "content": "G = (V, E, R) is a typed directed multigraph with |R| ≥ 2."
    },
    {
        "type": "Assumption",
        "number": "4.2",
        "name": "Encoder Expressivity",
        "content": "Enc : Q ∪ R → R^d satisfies E[q^⊤e_{r+}] > E[q^⊤e_{r−}] for r+ ∈ R*_l(Q), r− ∉ R*_l(Q)."
    },
    {
        "type": "Assumption",
        "number": "4.3",
        "name": "Depth Identifiability",
        "content": "The hop depth l of every (h, r, t) ∈ E0 is computable in O(|E0|) by BFS from S(Q)."
    },
    {
        "type": "Assumption",
        "number": "4.4",
        "name": "Path Decomposability",
        "content": "Every gold path P* = (e1, ..., eL) satisfies e_l ∈ E_l."
    },
    {
        "type": "Definition",
        "number": "4.5",
        "name": "Path-Coherence Score",
        "content": "PCS(Q, Efilt) = |{P ∈ P*(Q) : P ⊆ Efilt}| / |P*(Q)|; PCS = (1/|D|) * sum_{(Q,Efilt)∈D} PCS(Q, Efilt). Unlike edge-level F1, PCS is zero if any single edge of any gold path is pruned."
    },
    {
        "type": "Theorem",
        "number": "6.1",
        "name": "Path-Coherence Lower Bound",
        "content": "Let α_l = P[(h, r, t) ∈ Efilt | (h, r, t) ∈ El, r ∈ R*_l(Q)]. Under Assumption 4.4 and conditional independence of filtering across depths: E[PCS(Q, Efilt)] ≥ product_{l=1}^L α_l."
    },
    {
        "type": "Theorem",
        "number": "6.2",
        "name": "Structural Inferiority",
        "content": "Let α_l* (resp. α˜_l) be the per-depth TPR of the optimal depth-aware (resp. any depth-agnostic) filter. If I(Y;L|Q,R) > 0, then sum_{l} α˜_l < sum_{l} α_l* and ∃ l† with α˜_l† < α_l†*."
    },
    {
        "type": "Corollary",
        "number": "6.3",
        "name": "Multiplicative PCS Gap",
        "content": "E[PCS_agnostic] ≤ product_{l} α˜_l < product_{l} α_l* ≤ E[PCS_DSRQS]."
    },
    {
        "type": "Proposition",
        "number": "6.4",
        "name": "DC-Loss Drives Depth Separation",
        "content": "Let Δ = g(Q, r, l+) − g(Q, r, l−) with Δ < γ. Under gradient descent (step η) on LDC, Δ increases by 2η per step until Δ ≥ γ."
    },
]


def print_all():
    import sys
    def safe_print(text):
        # Force safe printing on any platform
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback: replace any problematic characters
            safe_text = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(safe_text)
    
    print("=" * 100)
    print("Paper Definitions and Theorems")
    print("=" * 100)
    for item in DEFINITIONS:
        print()
        safe_print(f"{item['type']} {item['number']}: {item['name']}")
        print("-" * 80)
        safe_print(item['content'])
    print()
    print("=" * 100)


if __name__ == "__main__":
    print_all()
