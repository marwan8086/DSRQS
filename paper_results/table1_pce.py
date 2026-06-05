# =============================================================================
# Paper Results: Table 1 - Depth-Conditional Relevance Shift
# =============================================================================

TABLE_1_DATA = {
    "Intent classes": [
        {
            "intent": "Etiology",
            "n": 54,
            "both_hops": "8.3", "both_hops_std": 1.9,
            "shift": 47.2, "shift_std": 8.1,
            "pce_exp": 22.4, "pce_exp_std": 5.3
        },
        {
            "intent": "Treatment",
            "n": 48,
            "both_hops": "7.1", "both_hops_std": 2.1,
            "shift": 31.4, "shift_std": 7.4,
            "pce_exp": 18.3, "pce_exp_std": 4.9
        },
        {
            "intent": "Phenotype",
            "n": 56,
            "both_hops": "9.0", "both_hops_std": 2.3,
            "shift": 60.9, "shift_std": 9.2,
            "pce_exp": 24.7, "pce_exp_std": 6.1
        },
        {
            "intent": "Gene-Func.",
            "n": 42,
            "both_hops": "7.6", "both_hops_std": 1.8,
            "shift": 53.1, "shift_std": 8.7,
            "pce_exp": 21.8, "pce_exp_std": 5.8
        },
        {
            "intent": "All",
            "n": 200,
            "both_hops": "8.0", "both_hops_std": 2.1,
            "shift": 49.3, "shift_std": 9.4,
            "pce_exp": 21.9, "pce_exp_std": 5.8
        }
    ],
    "notes": "Cohen's κ = 0.81"
}


def format_table_latex() -> str:
    """Generate LaTeX for Table 1."""
    latex = r"""
\begin{table}[t]
\centering
\caption{Depth-conditional relevance shift (200 annotated queries). Shift: fraction of relation types at both hops that flip label. PCE-Exp.: fraction of retrieved edges subject to conflation. Cohen's $\kappa = 0.81$.}
\label{tab:depth_shift}
\begin{tabular}{lcccccc}
\toprule
Intent & $n$ & \multicolumn{2}{c}{Both Hops} & \multicolumn{2}{c}{Shift (\%)} & \multicolumn{2}{c}{PCE-Exp. (\%)} \\
\midrule
Etiology & 54 & 8.3 & $\pm$1.9 & 47.2 & $\pm$8.1 & 22.4 & $\pm$5.3 \\
Treatment & 48 & 7.1 & $\pm$2.1 & 31.4 & $\pm$7.4 & 18.3 & $\pm$4.9 \\
Phenotype & 56 & 9.0 & $\pm$2.3 & 60.9 & $\pm$9.2 & 24.7 & $\pm$6.1 \\
Gene-Func. & 42 & 7.6 & $\pm$1.8 & 53.1 & $\pm$8.7 & 21.8 & $\pm$5.8 \\
All & 200 & 8.0 & $\pm$2.1 & 49.3 & $\pm$9.4 & 21.9 & $\pm$5.8 \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def print_table():
    """Print Table 1 readable format."""
    print("=" * 100)
    print("Table 1: Depth-Conditional Relevance Shift")
    print("=" * 100)
    print("Cohen's κ = 0.81")
    print("-" * 100)
    print(f"{'Intent':<15} {'n':<6} {'Both Hops':<15} {'Shift (%)':<15} {'PCE-Exp. (%)':<15}")
    print("-" * 100)

    for row in TABLE_1_DATA["Intent classes"]:
        both_hops_str = f"{row['both_hops']}±{row['both_hops_std']}"
        shift_str = f"{row['shift']}±{row['shift_std']}"
        pce_str = f"{row['pce_exp']}±{row['pce_exp_std']}"
        print(f"{row['intent']:<15} {row['n']:<6} {both_hops_str:<15} {shift_str:<15} {pce_str:<15}")
    print("=" * 100)


if __name__ == "__main__":
    print_table()
    print("\nLaTeX Code:")
    print("=" * 100)
    print(format_table_latex())
