# =============================================================================
# Appendix D: Additional Qualitative Example
# =============================================================================
from .utils import safe_print


APPENDIX_D = {
    "title": "Appendix D: Additional Qualitative Example",
    "query": {
        "intent": "OMIM-Hop3, Phenotype Intent",
        "text": "What facial dysmorphisms are associated with the allelic variant underlying Smith-Lemli-Opitz syndrome?"
    },
    "gold_path": [
        "1. Smith-Lemli-Opitz → (causal_gene) → DHCR7",
        "2. DHCR7 → (allelic_variant_of) → R352W",
        "3. R352W → (has_phenotype) → Microcephaly, Ptosis, Anteverted nares"
    ],
    "dsrqs_behavior": [
        "Hop 1 (causal_gene): Score = 0.92 > 0.5 → Retained",
        "Hop 2 (allelic_variant_of): Score = 0.88 > 0.5 → Retained (Depth-agnostic B5 scored this = 0.44 and incorrectly pruned it, breaking the chain)",
        "Hop 3 (has_phenotype): Score = 0.79 > 0.5 → Retained"
    ],
    "takeaway": "This example highlights how DSRQS preserves the critical allelic variant link (hop 2) which is frequently pruned by depth-agnostic filters due to its rarity at shallow depths but high relevance in deeper diagnostic reasoning chains."
}


def print_appendix_d():
    safe_print("=" * 120)
    safe_print(APPENDIX_D["title"])
    safe_print("=" * 120)
    safe_print("")

    safe_print("Query:")
    safe_print(f"  Intent: {APPENDIX_D['query']['intent']}")
    safe_print(f"  Text: {APPENDIX_D['query']['text']}")
    safe_print("")

    safe_print("Gold Path (3-hop):")
    for step in APPENDIX_D["gold_path"]:
        safe_print(f"  {step}")
    safe_print("")

    safe_print("DSRQS Filtering Behavior:")
    for line in APPENDIX_D["dsrqs_behavior"]:
        safe_print(f"  • {line}")
    safe_print("")

    safe_print("=" * 120)
    safe_print(APPENDIX_D["takeaway"])
    safe_print("=" * 120)


def get_latex():
    latex = r"""
% =============================================================================
% Appendix D: Additional Qualitative Example
% =============================================================================
\section*{Appendix D: Additional Qualitative Example}

\paragraph{Query (OMIM-Hop3, Phenotype Intent):}
\begin{quote}
What facial dysmorphisms are associated with the allelic variant underlying Smith-Lemli-Opitz syndrome?
\end{quote}

\paragraph{Gold Path (3-hop):}
\begin{enumerate}
    \item $\text{Smith-Lemli-Opitz} \xrightarrow{\text{causal\_gene}} \text{DHCR7}$
    \item $\text{DHCR7} \xrightarrow{\text{allelic\_variant\_of}} \text{R352W}$
    \item $\text{R352W} \xrightarrow{\text{has\_phenotype}} \text{Microcephaly, Ptosis, Anteverted nares}$
\end{enumerate}

\paragraph{DSRQS Filtering Behavior:}
\begin{itemize}
    \item Hop 1 (causal\_gene): Score 0.92 > 0.5 → Retained.
    \item Hop 2 (allelic\_variant\_of): Score 0.88 > 0.5 → Retained. (Depth-agnostic B5 scored this 0.44 and incorrectly pruned it, breaking the chain.)
    \item Hop 3 (has\_phenotype): Score 0.79 > 0.5 → Retained.
\end{itemize}

This example highlights how DSRQS preserves the critical allelic variant link (hop 2) which is frequently pruned by depth-agnostic filters due to its rarity at shallow depths but high relevance in deeper diagnostic reasoning chains.
"""
    return latex


if __name__ == "__main__":
    print_appendix_d()
    print("\n\nLaTeX Code:\n")
    print(get_latex())
