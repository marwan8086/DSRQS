# =============================================================================
# Appendix B: OMIM-Hop3 Benchmark
# =============================================================================
from .utils import safe_print


APPENDIX_B = {
    "title": "Appendix B: OMIM-Hop3 Benchmark",
    "data_source": "OMIM API (academic licence, December 2025)",
    "selection_criteria": [
        "(i) rare disease (phenotype type 3);",
        "(ii) at least one allelic variant;",
        "(iii) phenotypic-series membership;",
        "(iv) gene entry with GO pathway annotation."
    ],
    "exclusions": "Entries with any path of length <= 2 in the OMIM KG were excluded.",
    "annotation": "Two annotators verified the 3-hop requirement independently (κ = 0.82); adjudicated by a third expert.",
    "final_stats": "183 entries from 79 phenotypic series.",
    "example": {
        "template": "Through which cellular pathway does the protein product of the causal gene of [disease] exert its pathogenic effect?",
        "required_chain": [
            "disease → causal_gene →",
            "encodes_protein → functions_in_pathway."
        ]
    }
}


def print_appendix_b():
    safe_print("=" * 120)
    safe_print(APPENDIX_B["title"])
    safe_print("=" * 120)
    safe_print("")
    safe_print(f"Data Source: {APPENDIX_B['data_source']}")
    safe_print("")
    safe_print("Selection Criteria:")
    for criterion in APPENDIX_B["selection_criteria"]:
        safe_print(f"  - {criterion}")
    safe_print("")
    safe_print(f"Exclusions: {APPENDIX_B['exclusions']}")
    safe_print("")
    safe_print(f"Annotation Process: {APPENDIX_B['annotation']}")
    safe_print("")
    safe_print(f"Final Statistics: {APPENDIX_B['final_stats']}")
    safe_print("")
    safe_print("Example Question Template (Gene-Function Intent):")
    safe_print(f"  {APPENDIX_B['example']['template']}")
    safe_print("")
    safe_print("Required Gold Chain:")
    for step in APPENDIX_B['example']['required_chain']:
        safe_print(f"  {step}")
    safe_print("")
    safe_print("=" * 120)


def get_latex():
    latex = r"""
% =============================================================================
% Appendix B: OMIM-Hop3 Benchmark
% =============================================================================
\section*{Appendix B: OMIM-Hop3 Benchmark}

OMIM API (academic licence, December 2025).

Selection:
\begin{itemize}
    \item rare disease (phenotype type 3);
    \item at least one allelic variant;
    \item phenotypic-series membership;
    \item gene entry with GO pathway annotation.
\end{itemize}

Exclusions: Entries with any path of length $\leq 2$ in the OMIM KG were excluded.

Annotation: Two annotators verified the 3-hop requirement independently (Cohen's $\kappa = 0.82$); adjudicated by a third expert.

Final: $183$ entries from $79$ phenotypic series.

\paragraph{Example Question Template (Gene-Function Intent):}
\begin{quote}
Through which cellular pathway does the protein product of the causal gene of [disease] exert its pathogenic effect?
\end{quote}

\paragraph{Required Gold Chain:}
\[
\text{disease} \xrightarrow{\text{causal\_gene}} \quad
\xrightarrow{\text{encodes\_protein}} \xrightarrow{\text{functions\_in\_pathway}}
\]
"""
    return latex


if __name__ == "__main__":
    print_appendix_b()
    print("\n\nLaTeX Code:\n")
    print(get_latex())
