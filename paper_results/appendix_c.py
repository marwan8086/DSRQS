# =============================================================================
# Appendix C: Reproducibility Checklist
# =============================================================================
from .utils import safe_print


APPENDIX_C = {
    "title": "Appendix C: Reproducibility Checklist",
    "sections": {
        "All articles": [
            "Claims stated: [yes]",
            "Claims substantiated: [yes]",
            "Assumptions stated: [yes]",
            "Pseudocode: [yes]",
            "Design choices justified: [yes]"
        ],
        "Theoretical": [
            "Conditions stated: [yes]",
            "Proofs: [yes, Appendix A]",
            "Corollaries: [yes]"
        ],
        "Computational": [
            "Code (URL on acceptance): [yes]",
            "MIT licence: [yes]",
            "Datasets released: [yes]",
            "Seeds documented: [yes]",
            "Hardware (A100, 80 GB): [yes]",
            "Metrics defined: [yes]",
            "5-fold CV × 5 seeds: [yes]",
            "No cherry-picking: [yes]",
            "Std reported: [yes]",
            "Hyperparameters (λ=0.4, γ=0.25, ρ=16, θ=0.5, lr=5e-4): [yes]",
            "Wilcoxon tests: [yes]"
        ],
        "Datasets": [
            "Three new datasets released: [yes]",
            "Licences (Orphanet CC-BY-4.0; DisGeNET CC-BY-NC-SA-4.0; OMIM academic): [yes]",
            "Sources cited: [yes]",
            "Preprocessing: [yes]",
            "Cohen's κ (0.79–0.82): [yes]"
        ]
    },
    "extended_env": {
        "OS": "Ubuntu 22.04.3 LTS",
        "Python": "3.10.13",
        "PyTorch": "2.1.2 with CUDA 12.1",
        "Transformers": "4.36.2",
        "DGL": "2.1.0 (for graph operations)",
        "Scikit-learn": "1.3.2 (for metrics)",
        "GPU": "single NVIDIA A100 80GB",
        "Driver": "535.129.03",
        "Total computation time": "18.4 GPU-hours"
    },
    "hyperparam_sensitivity": {
        "gamma": {
            "values": [0.10, 0.25, 0.50],
            "pcs": [0.787, 0.801, 0.795]
        },
        "learning_rate": {
            "values": ["1e-4", "5e-4", "1e-3"],
            "optimal": "5e-4"
        }
    }
}


def print_appendix_c():
    safe_print("=" * 120)
    safe_print(APPENDIX_C["title"])
    safe_print("=" * 120)
    safe_print("")

    for section, items in APPENDIX_C["sections"].items():
        safe_print(f"  {section}:")
        for item in items:
            safe_print(f"    - {item}")
        safe_print("")

    safe_print("=" * 120)
    safe_print("Extended Implementation Details:")
    safe_print("=" * 120)
    for key, value in APPENDIX_C["extended_env"].items():
        safe_print(f"  {key}: {value}")
    safe_print("")
    safe_print("=" * 120)
    safe_print("Hyperparameter Sensitivity:")
    safe_print("=" * 120)
    safe_print("")
    safe_print("Margin γ on DisGeNET-RD411 validation split:")
    for g, p in zip(APPENDIX_C["hyperparam_sensitivity"]["gamma"]["values"], APPENDIX_C["hyperparam_sensitivity"]["gamma"]["pcs"]):
        safe_print(f"  γ = {g}  →  PCS = {p}")
    safe_print("")
    safe_print("Learning Rate Tuning:")
    for lr in APPENDIX_C["hyperparam_sensitivity"]["learning_rate"]["values"]:
        opt_str = " (OPTIMAL)" if lr == APPENDIX_C["hyperparam_sensitivity"]["learning_rate"]["optimal"] else ""
        safe_print(f"  lr = {lr}{opt_str}")
    safe_print("")
    safe_print("=" * 120)


def get_latex():
    latex = r"""
% =============================================================================
% Appendix C: Reproducibility Checklist
% =============================================================================
\section*{Appendix C: Reproducibility Checklist}

\subsection*{All articles}
\begin{itemize}
    \item Claims stated: [yes]
    \item Claims substantiated: [yes]
    \item Assumptions stated: [yes]
    \item Pseudocode: [yes]
    \item Design choices justified: [yes]
\end{itemize}

\subsection*{Theoretical}
\begin{itemize}
    \item Conditions stated: [yes]
    \item Proofs: [yes, Appendix A]
    \item Corollaries: [yes]
\end{itemize}

\subsection*{Computational}
\begin{itemize}
    \item Code (URL on acceptance): [yes]
    \item MIT licence: [yes]
    \item Datasets released: [yes]
    \item Seeds documented: [yes]
    \item Hardware (A100, 80 GB): [yes]
    \item Metrics defined: [yes]
    \item 5-fold CV × 5 seeds: [yes]
    \item No cherry-picking: [yes]
    \item Standard deviation reported: [yes]
    \item Hyperparameters:
          \begin{itemize}
            \item $\lambda = 0.4$
            \item $\gamma = 0.25$
            \item $\rho = 16$
            \item $\theta = 0.5$
            \item Learning rate $\text{lr} = 5 \times 10^{-4}$
          \end{itemize}
    \item Wilcoxon tests: [yes]
\end{itemize}

\subsection*{Datasets}
\begin{itemize}
    \item Three new datasets released: [yes]
    \item Licences:
        \begin{itemize}
            \item Orphanet: CC-BY-4.0
            \item DisGeNET: CC-BY-NC-SA-4.0
            \item OMIM: Academic
        \end{itemize}
    \item Sources cited: [yes]
    \item Preprocessing: [yes]
    \item Inter-annotator agreement: Cohen's $\kappa$ in (0.79–0.82): [yes]
\end{itemize}

\subsection*{Extended Implementation Details}
For full reproducibility, we document the exact software environment:
\begin{itemize}
    \item \textbf{OS}: Ubuntu 22.04.3 LTS
    \item \textbf{Python}: 3.10.13
    \item \textbf{PyTorch}: 2.1.2 with CUDA 12.1
    \item \textbf{Transformers}: 4.36.2
    \item \textbf{DGL}: 2.1.0 (for graph operations)
    \item \textbf{Scikit-learn}: 1.3.2 (for metrics)
\end{itemize}
All experiments used a single NVIDIA A100 80GB GPU with driver version 535.129.03.
Total computation time for the full benchmark suite (including all baselines and 5-fold CV) was approximately 18.4 GPU-hours.

\subsection*{Hyperparameter Sensitivity}
We further analyzed sensitivity to the margin parameter $\gamma$ in $L_{\text{DC}}$.
On DisGeNET-RD411 validation split,
PCS varied as $\gamma \in \{0.10, 0.25, 0.50\}$ produced PCS of $0.787$, $0.801$, and $0.795$ respectively,
confirming robustness near the chosen $\gamma = 0.25$.
The learning rate was also tuned over $\{1 \times 10^{-4}, 5 \times 10^{-4}, 1 \times 10^{-3}\}$,
with $5 \times 10^{-4}$ yielding optimal convergence.
"""
    return latex


if __name__ == "__main__":
    print_appendix_c()
    print("\n\nLaTeX Code:\n")
    print(get_latex())
