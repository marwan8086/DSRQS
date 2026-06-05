# =============================================================================
# Paper Results: Table 3 - Main Results
# =============================================================================
from typing import Dict, List

TABLE_3_DATA = {
    "orphanet_fq274": {
        "B1 (No Filter)": {
            "PCS": 0.415, "PCS_std": None,
            "Fe1": None, "Fe1_std": None,
            "H": 18.7, "H_std": None,
            "Fa1": 0.584, "Fa1_std": None
        },
        "B2 (Heuristic)": {
            "PCS": 0.526, "PCS_std": None,
            "Fe1": 0.618, "Fe1_std": None,
            "H": 15.9, "H_std": None,
            "Fa1": 0.621, "Fa1_std": None
        },
        "B3 (Cosine)": {
            "PCS": 0.573, "PCS_std": None,
            "Fe1": 0.671, "Fe1_std": None,
            "H": 13.8, "H_std": None,
            "Fa1": 0.643, "Fa1_std": None
        },
        "B4 (Bilinear)": {
            "PCS": 0.624, "PCS_std": None,
            "Fe1": 0.716, "Fe1_std": None,
            "H": 11.7, "H_std": None,
            "Fa1": 0.672, "Fa1_std": None
        },
        "B5 (Bilinear-SupCon)": {
            "PCS": 0.646, "PCS_std": None,
            "Fe1": 0.745, "Fe1_std": None,
            "H": 10.5, "H_std": None,
            "Fa1": 0.691, "Fa1_std": None
        },
        "B6 (No DC)": {
            "PCS": 0.719, "PCS_std": None,
            "Fe1": 0.792, "Fe1_std": None,
            "H": 9.3, "H_std": None,
            "Fa1": 0.727, "Fa1_std": None
        },
        "DSRQS": {
            "PCS": 0.738, "PCS_std": 0.021,
            "Fe1": 0.818, "Fe1_std": None,
            "H": 7.8, "H_std": None,
            "Fa1": 0.749, "Fa1_std": None,
            "significance": "‡"
        }
    },
    "disgenet_rd411": {
        "B1 (No Filter)": {
            "PCS": 0.392, "PCS_std": None,
            "Fe1": None, "Fe1_std": None,
            "H": 20.5, "H_std": None,
            "Fa1": 0.571, "Fa1_std": None
        },
        "B2 (Heuristic)": {
            "PCS": 0.507, "PCS_std": None,
            "Fe1": 0.594, "Fe1_std": None,
            "H": 17.8, "H_std": None,
            "Fa1": 0.602, "Fa1_std": None
        },
        "B3 (Cosine)": {
            "PCS": 0.548, "PCS_std": None,
            "Fe1": 0.641, "Fe1_std": None,
            "H": 15.9, "H_std": None,
            "Fa1": 0.625, "Fa1_std": None
        },
        "B4 (Bilinear)": {
            "PCS": 0.601, "PCS_std": None,
            "Fe1": 0.695, "Fe1_std": None,
            "H": 13.3, "H_std": None,
            "Fa1": 0.651, "Fa1_std": None
        },
        "B5 (Bilinear-SupCon)": {
            "PCS": 0.658, "PCS_std": None,
            "Fe1": 0.763, "Fe1_std": None,
            "H": 11.9, "H_std": None,
            "Fa1": 0.673, "Fa1_std": None
        },
        "B6 (No DC)": {
            "PCS": 0.742, "PCS_std": None,
            "Fe1": 0.814, "Fe1_std": None,
            "H": 10.1, "H_std": None,
            "Fa1": 0.716, "Fa1_std": None
        },
        "DSRQS": {
            "PCS": 0.768, "PCS_std": 0.018,
            "Fe1": 0.837, "Fe1_std": None,
            "H": 8.4, "H_std": None,
            "Fa1": 0.743, "Fa1_std": None,
            "significance": "‡"
        }
    },
    "omim_hop3": {
        "B1 (No Filter)": {
            "PCS": 0.298, "PCS_std": None,
            "Fe1": None, "Fe1_std": None,
            "H": 25.1, "H_std": None,
            "Fa1": 0.519, "Fa1_std": None
        },
        "B2 (Heuristic)": {
            "PCS": 0.423, "PCS_std": None,
            "Fe1": 0.491, "Fe1_std": None,
            "H": 22.4, "H_std": None,
            "Fa1": 0.556, "Fa1_std": None
        },
        "B3 (Cosine)": {
            "PCS": 0.472, "PCS_std": None,
            "Fe1": 0.553, "Fe1_std": None,
            "H": 20.6, "H_std": None,
            "Fa1": 0.581, "Fa1_std": None
        },
        "B4 (Bilinear)": {
            "PCS": 0.527, "PCS_std": None,
            "Fe1": 0.612, "Fe1_std": None,
            "H": 18.1, "H_std": None,
            "Fa1": 0.608, "Fa1_std": None
        },
        "B5 (Bilinear-SupCon)": {
            "PCS": 0.588, "PCS_std": None,
            "Fe1": 0.676, "Fe1_std": None,
            "H": 16.5, "H_std": None,
            "Fa1": 0.629, "Fa1_std": None
        },
        "B6 (No DC)": {
            "PCS": 0.681, "PCS_std": None,
            "Fe1": 0.742, "Fe1_std": None,
            "H": 13.1, "H_std": None,
            "Fa1": 0.672, "Fa1_std": None
        },
        "DSRQS": {
            "PCS": 0.714, "PCS_std": 0.026,
            "Fe1": 0.769, "Fe1_std": None,
            "H": 10.9, "H_std": None,
            "Fa1": 0.699, "Fa1_std": None,
            "significance": "†"
        }
    }
}

DATASET_NAMES = {
    "orphanet_fq274": "Orphanet-FQ274",
    "disgenet_rd411": "DisGeNET-RD411",
    "omim_hop3": "OMIM-Hop3"
}


def format_table_latex() -> str:
    """Generate LaTeX code for Table 3."""
    latex = r"""
\begin{table*}[t]
\centering
\caption{Main results (5-fold CV $\times$ 5 seeds). PCS: Path Coherence Score (primary metric). Fe1: Edge F1. H: Hallucination rate (\%). Fa1: Answer F1. T: Latency (ms/edge). Significance vs. B5: $\dagger p<0.05$, $\ddagger p<0.01$ (Wilcoxon signed-rank test). Best per column in bold.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Orphanet-FQ274} & \multicolumn{3}{c}{DisGeNET-RD411} & \multicolumn{3}{c}{OMIM-Hop3} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Model & PCS & H & Fa1 & PCS & H & Fa1 & PCS & H & Fa1 \\
\midrule
"""
    models = list(TABLE_3_DATA["orphanet_fq274"].keys())
    
    for model in models:
        row = f"{model} "
        for dataset in ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]:
            data = TABLE_3_DATA[dataset][model]
            
            # PCS with std
            pcs_str = f"{data['PCS']:.3f}"
            if data['PCS_std'] is not None:
                pcs_str += f"\\std{{{data['PCS_std']:.3f}}}"
            if data.get('significance'):
                pcs_str += f"^{data['significance']}"
            if model == "DSRQS":
                pcs_str = f"\\textbf{{{pcs_str}}}"
            
            # H
            h_str = f"{data['H']:.1f}"
            if model == "DSRQS":
                h_str = f"\\textbf{{{h_str}}}"
            
            # Fa1
            fa1_str = f"{data['Fa1']:.3f}"
            if model == "DSRQS":
                fa1_str = f"\\textbf{{{fa1_str}}}"
            
            row += f"& {pcs_str} & {h_str} & {fa1_str} "
        
        row += r"\\"
        latex += row + "\n"
    
    latex += r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    return latex


def print_table():
    """Print Table 3 in readable format."""
    print("=" * 140)
    print("Table 3: Main Results")
    print("=" * 140)
    print()
    
    for dataset in ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]:
        print(f"{DATASET_NAMES[dataset]}")
        print("-" * 140)
        print(f"{'Model':<20} {'PCS':<10} {'H (%)':<10} {'Fa1':<10}")
        print("-" * 140)
        
        for model, data in TABLE_3_DATA[dataset].items():
            pcs_str = f"{data['PCS']:.3f}"
            if data.get('significance'):
                pcs_str += data['significance']
            if model == "DSRQS":
                pcs_str = f"**{pcs_str}**"
            
            print(f"{model:<20} {pcs_str:<10} {data['H']:<10.1f} {data['Fa1']:<10.3f}")
        
        print()
    
    print("=" * 140)
    print("† p<0.05, ‡ p<0.01 vs. B5 (Wilcoxon)")
    print("=" * 140)


if __name__ == "__main__":
    print_table()
    print()
    print("\nLaTeX Code:")
    print("=" * 80)
    print(format_table_latex())
