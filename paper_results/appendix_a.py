# =============================================================================
# Appendix A: Complete Proofs
# =============================================================================
from .utils import safe_print


APPENDIX_A = {
    "title": "Appendix A: Complete Proofs",
    "sections": [
        {
            "name": "Theorem 6.1 (Path-Coherence Lower Bound)",
            "proof": r"""
For path \( P_j^* = (e_{j,1}, \dots, e_{j,L}) \) with \( e_{j,\ell} \in E_\ell \) (Assumption 4.4),
conditional independence gives:
\[
\mathbb{P}[P_j^* \subseteq E_{\text{filt}}] = \prod_{\ell} \mathbb{P}[e_{j,\ell} \in E_{\text{filt}}] \geq \prod_{\ell} \alpha_\ell.
\]
Averaging over paths and queries yields Eq. (4).
""",
            "simple": """
For path P_j* = (e_j1, ..., ejL) with e_jℓ ∈ Eℓ (Assumption 4.4),
conditional independence gives:
P[P_j* ⊆ E_filt] = product over ℓ of P[e_jℓ ∈ E_filt] ≥ product over ℓ of αℓ.
Averaging over paths and queries yields Equation 4.
"""
        },
        {
            "name": "Theorem 6.2 (Structural Inferiority)",
            "proof": r"""
Any depth-agnostic filter classifies on \( \tilde{y} = \mathbb{E}_L[y_L] \).
When \( I(Y; L | Q, R) > 0 \),
\[
H(Y | Q, R, L) < H(Y | Q, R)
\]
by data-processing inequality; the mapping \( \{y_\ell\} \mapsto \tilde{y} \) is strictly lossy.
The Bayes classifier on \( \tilde{y} \) has strictly higher error at some \( \ell^\dagger \).
""",
            "simple": """
Any depth-agnostic filter classifies on y-tilde = E_L[y_L].
When I(Y; L | Q, R) > 0,
H(Y | Q, R, L) < H(Y | Q, R) by data-processing inequality;
the mapping {y_ℓ} -> y-tilde is strictly lossy.
The Bayes classifier on y-tilde has strictly higher error at some ℓ†.
"""
        },
        {
            "name": "Proposition 6.4 (DC-Loss Drives Depth Separation)",
            "proof": r"""
\[
L_{\text{DC}} = \gamma - \Delta > 0;
\]
sub-gradients -1 and +1;
\[
\Delta^{(t+1)} = \Delta^{(t)} + 2 \eta.
\]
""",
            "simple": """
L_DC = gamma - Delta > 0;
sub-gradients -1 and +1;
Delta(t+1) = Delta(t) + 2 eta.
"""
        }
    ]
}


def print_appendix_a():
    safe_print("=" * 120)
    safe_print(APPENDIX_A["title"])
    safe_print("=" * 120)

    for section in APPENDIX_A["sections"]:
        safe_print("")
        safe_print(f"  {section['name']}")
        safe_print("  " + "-" * 100)
        safe_print(section['simple'])
        safe_print("")

    safe_print("=" * 120)


def get_latex():
    latex = r"""
% =============================================================================
% Appendix A: Complete Proofs
% =============================================================================
\section*{Appendix A: Complete Proofs}

\subsection*{Theorem 6.1 (Path-Coherence Lower Bound)}
For path \( P_j^* = (e_{j,1}, \dots, e_{j,L}) \) with \( e_{j,\ell} \in E_\ell \) (Assumption 4.4),
conditional independence gives:
\[
\mathbb{P}[P_j^* \subseteq E_{\text{filt}}] = \prod_{\ell} \mathbb{P}[e_{j,\ell} \in E_{\text{filt}}] \geq \prod_{\ell} \alpha_\ell.
\]
Averaging over paths and queries yields Eq. (4).

\subsection*{Theorem 6.2 (Structural Inferiority)}
Any depth-agnostic filter classifies on \( \tilde{y} = \mathbb{E}_L[y_L] \).
When \( I(Y; L | Q, R) > 0 \),
\[
H(Y | Q, R, L) < H(Y | Q, R)
\]
by data-processing inequality; the mapping \( \{y_\ell\} \mapsto \tilde{y} \) is strictly lossy.
The Bayes classifier on \( \tilde{y} \) has strictly higher error at some \( \ell^\dagger \).

\subsection*{Proposition 6.4 (DC-Loss Drives Depth Separation)}
\[
L_{\text{DC}} = \gamma - \Delta > 0;
\]
sub-gradients -1 and +1;
\[
\Delta^{(t+1)} = \Delta^{(t)} + 2 \eta.
\]
"""
    return latex


if __name__ == "__main__":
    print_appendix_a()
    print("\n\nLaTeX Code:\n")
    print(get_latex())
