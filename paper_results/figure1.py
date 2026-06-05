# =============================================================================
# Figure 1: Estimated mutual information I(Y;L|Q,R)
# =============================================================================
import numpy as np

FIGURE_1 = {
    "intents": ["Gene-Func.", "Phenotype", "Treatment", "Etiology"],
    "relations": ["expr_in", "has_phen.", "treats", "causal_mut", "gene_dis.", "pathway", "allelic_var", "series"],
    "values": np.array([
        [0.19, 0.07, 0.08, 0.22, 0.31, 0.26, 0.18, 0.11],
        [0.28, 0.33, 0.05, 0.08, 0.14, 0.11, 0.07, 0.19],
        [0.09, 0.03, 0.27, 0.11, 0.06, 0.22, 0.13, 0.07],
        [0.21, 0.04, 0.03, 0.31, 0.08, 0.19, 0.28, 0.14]
    ]),
    "estimate": {
        "point": 0.173,
        "std": 0.041,
        "ci95_lower": 0.093,
        "ci95_upper": 0.251,
    }
}


def print_figure():
    print("=" * 100)
    print("Figure 1: Estimated mutual information I(Y;L|Q,R)")
    print("=" * 100)
    print()
    print(f"Point estimate: I(Y;L|Q,R) = {FIGURE_1['estimate']['point']} ± {FIGURE_1['estimate']['std']} bits")
    print(f"95% CI: [{FIGURE_1['estimate']['ci95_lower']}, {FIGURE_1['estimate']['ci95_upper']}]")
    print()
    print("Per-cell values (Intent × Relation Type):")
    print("-" * 100)
    for i, intent in enumerate(FIGURE_1["intents"]):
        values_str = "  ".join([f"{v:.2f}" for v in FIGURE_1["values"][i]])
        print(f"{intent:>12}:  {values_str}")
    print()
    print("=" * 100)


if __name__ == "__main__":
    print_figure()
