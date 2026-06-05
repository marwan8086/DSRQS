# =============================================================================
# Paper Results: Table 4 - Per-Depth TPR on DisGeNET-RD411
# =============================================================================

TABLE_4_DATA = {
    "B4": {
        "alpha_1": 0.841, "alpha_1_std": 0.027,
        "alpha_2": 0.712, "alpha_2_std": 0.038,
        "delta_alpha": 0.129,
        "alpha_product": 0.599,
        "PCS": 0.598
    },
    "B5": {
        "alpha_1": 0.853, "alpha_1_std": 0.024,
        "alpha_2": 0.773, "alpha_2_std": 0.034,
        "delta_alpha": 0.080,
        "alpha_product": 0.659,
        "PCS": 0.659
    },
    "B6": {
        "alpha_1": 0.871, "alpha_1_std": 0.021,
        "alpha_2": 0.854, "alpha_2_std": 0.028,
        "delta_alpha": 0.017,
        "alpha_product": 0.744,
        "PCS": 0.744
    },
    "DSRQS": {
        "alpha_1": 0.891, "alpha_1_std": 0.018,
        "alpha_2": 0.899, "alpha_2_std": 0.022,
        "delta_alpha": 0.008,
        "alpha_product": 0.801,
        "PCS": 0.801
    }
}


def format_table_latex() -> str:
    """Generate LaTeX code for Table 4."""
    latex = r"""
\begin{table}[t]
\centering
\caption{Per-depth TPR on DisGeNET-RD411. $\Delta\alpha = |\alpha_1 - \alpha_2|$: depth imbalance (lower is better). Note that $\alpha_1 \cdot \alpha_2 \approx$ PCS, verifying Theorem 6.1.}
\label{tab:depth_imbalance}
\begin{tabular}{lcccccc}
\toprule
Model & $\alpha_1$ & $\alpha_2$ & $\Delta\alpha$ & $\alpha_1 \cdot \alpha_2$ & PCS \\
\midrule
"""
    for model, data in TABLE_4_DATA.items():
        a1 = f"{data['alpha_1']:.3f}\\std{{{data['alpha_1_std']:.3f}}}"
        a2 = f"{data['alpha_2']:.3f}\\std{{{data['alpha_2_std']:.3f}}}"
        delta = f"{data['delta_alpha']:.3f}"
        product = f"{data['alpha_product']:.3f}"
        pcs = f"{data['PCS']:.3f}"
        
        if model == "DSRQS":
            a1 = f"\\textbf{{{a1}}}"
            a2 = f"\\textbf{{{a2}}}"
            delta = f"\\textbf{{{delta}}}"
            product = f"\\textbf{{{product}}}"
            pcs = f"\\textbf{{{pcs}}}"
        
        latex += f"{model} & {a1} & {a2} & {delta} & {product} & {pcs} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def print_table():
    """Print Table 4 in readable format."""
    print("=" * 100)
    print("Table 4: Per-Depth TPR on DisGeNET-RD411")
    print("=" * 100)
    print()
    print(f"{'Model':<10} {'α₁':<15} {'α₂':<15} {'Δα':<10} {'α₁·α₂':<10} {'PCS':<10}")
    print("-" * 100)
    
    for model, data in TABLE_4_DATA.items():
        a1_str = f"{data['alpha_1']:.3f}±{data['alpha_1_std']:.3f}"
        a2_str = f"{data['alpha_2']:.3f}±{data['alpha_2_std']:.3f}"
        delta = f"{data['delta_alpha']:.3f}"
        product = f"{data['alpha_product']:.3f}"
        pcs = f"{data['PCS']:.3f}"
        
        if model == "DSRQS":
            a1_str = f"**{a1_str}**"
            a2_str = f"**{a2_str}**"
            delta = f"**{delta}**"
            product = f"**{product}**"
            pcs = f"**{pcs}**"
        
        print(f"{model:<10} {a1_str:<15} {a2_str:<15} {delta:<10} {product:<10} {pcs:<10}")
    
    print()
    print("=" * 100)
    print("Δα = |α₁ - α₂| - lower is better")
    print("=" * 100)


if __name__ == "__main__":
    print_table()
    print()
    print("\nLaTeX Code:")
    print("=" * 80)
    print(format_table_latex())
