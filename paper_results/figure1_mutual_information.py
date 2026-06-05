# =============================================================================
# Paper Results: Figure 1 - Mutual Information Visualization
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

FIGURE_1_DATA = {
    "intents": ["Gene-Func.", "Phenotype", "Treatment", "Etiology"],
    "relations": [
        "expr_in",
        "has_phen.",
        "treats",
        "causal_mut",
        "gene_dis.",
        "pathway",
        "allelic_var",
        "series"
    ],
    "values": np.array([
        [0.19, 0.07, 0.08, 0.22, 0.31, 0.26, 0.18, 0.11],
        [0.28, 0.33, 0.05, 0.08, 0.14, 0.11, 0.07, 0.19],
        [0.09, 0.03, 0.27, 0.11, 0.06, 0.22, 0.13, 0.07],
        [0.21, 0.04, 0.03, 0.31, 0.08, 0.19, 0.28, 0.14]
    ])
}


def plot_figure_1():
    """Plot Figure 1 from paper - heatmap of mutual information."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values = FIGURE_1_DATA["values"]
    intents = FIGURE_1_DATA["intents"]
    relations = FIGURE_1_DATA["relations"]
    
    im = ax.imshow(values, cmap='Blues', vmin=0, vmax=0.35)
    
    # Set ticks
    ax.set_xticks(np.arange(len(relations)))
    ax.set_yticks(np.arange(len(intents)))
    ax.set_xticklabels(relations, rotation=45, ha='right')
    ax.set_yticklabels(intents)
    
    # Annotate values on the heatmap
    for i in range(len(intents)):
        for j in range(len(relations)):
            ax.text(
                j, i, f"{values[i, j]:.2f}",
                ha='center', va='center',
                color='black', fontsize=10
            )
    
    ax.set_xlabel('Relation Type')
    ax.set_ylabel('Query-Intent Class')
    plt.colorbar(im, label=r"$\hat{I}(Y; L \mid Q, R)$ (bits)")
    
    plt.title("Figure 1: Estimated Mutual Information")
    plt.tight_layout()
    plt.savefig("paper_results/figure1.png")
    plt.savefig("paper_results/figure1.pdf")
    print("✅ Figure 1 saved to paper_results/figure1.pdf")
    plt.show()


def print_figure1():
    print("=" * 100)
    print("Figure 1: Estimated Mutual Information")
    print("=" * 100)
    print()
    print("Values (Intents x Relations):")
    print("-" * 100)
    
    for i, intent in enumerate(FIGURE_1_DATA["intents"]):
        line = f"{intent:>15}: "
        for v in FIGURE_1_DATA["values"][i]:
            line += f"{v:.2f}  "
        print(line)
    
    print()
    print("=" * 100)
    print("ˆI(Y; L | Q, R) = 0.173 ± 0.041 bits")
    print("95% CI: [0.093, 0.251]")
    print("=" * 100)


if __name__ == "__main__":
    print_figure1()
    print("\nCreating plot...")
    plot_figure_1()
