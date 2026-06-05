# =============================================================================
# Paper Results: Table 2 - Filter Error Decomposition
# =============================================================================

TABLE_2_DATA = {
    "filters": [
        {
            "name": "Cosine",
            "TP": 5841, "FP": 1972, "FN": 2103,
            "Fe1": 0.672,
            "pce_percent": 41.2, "pce_percent_std": 6.8
        },
        {
            "name": "Bilinear-BCE",
            "TP": 6314, "FP": 1588, "FN": 1630,
            "Fe1": 0.741,
            "pce_percent": 38.7, "pce_percent_std": 7.1
        },
        {
            "name": "Bilinear+SupCon",
            "TP": 6710, "FP": 1201, "FN": 1234,
            "Fe1": 0.803,
            "pce_percent": 35.4, "pce_percent_std": 6.3
        }
    ],
    "notes": "≈14,000 edges total"
}


def format_table_latex() -> str:
    """Generate LaTeX for Table2."""
    latex = r"""
\begin{table}[t]
\centering
\caption{Filter error decomposition on $\approx$14,000 edges. PCE-\%: fraction of (FP+FN) attributable to position conflation.}
\label{tab:error_decomposition}
\begin{tabular}{lcccccc}
\toprule
Filter & TP & FP & FN & Fe1 & \multicolumn{2}{c}{PCE-\%} \\
\midrule
Cosine & 5,841 & 1,972 & 2,103 & 0.672 & 41.2 & $\pm$6.8 \\
Bilinear-BCE & 6,314 & 1,588 & 1,630 & 0.741 & 38.7 & $\pm$7.1 \\
Bilinear+SupCon & 6,710 & 1,201 & 1,234 & 0.803 & 35.4 & $\pm$6.3 \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def print_table():
    """Print Table 2."""
    print("=" * 100)
    print("Table 2: Filter Error Decomposition")
    print("=" * 100)
    print("≈14,000 edges")
    print("-" * 100)
    print(f"{'Filter':<18} {'TP':<8} {'FP':<8} {'FN':<8} {'Fe1':<8} {'PCE-%':<12}")
    print("-" * 100)

    for filter_data in TABLE_2_DATA["filters"]:
        tp_str = f"{filter_data['TP']:,}"
        fp_str = f"{filter_data['FP']:,}"
        fn_str = f"{filter_data['FN']:,}"
        fe1_str = f"{filter_data['Fe1']:.3f}"
        pce_str = f"{filter_data['pce_percent']:.1f}±{filter_data['pce_percent_std']:.1f}"
        print(f"{filter_data['name']:<18} {tp_str:<8} {fp_str:<8} {fn_str:<8} {fe1_str:<8} {pce_str:<12}")
    
    print("=" * 100)


if __name__ == "__main__":
    print_table()
    print("\nLaTeX Code:")
    print("=" * 100)
    print(format_table_latex())
