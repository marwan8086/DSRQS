# =============================================================================
# Visualize demo results
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load results
with open("demo_results/demo_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
statistics = data["statistics"]

output_dir = Path("demo_results/plots")
output_dir.mkdir(exist_ok=True, parents=True)

print("Generating plots...")

# 1. Bar plot with confidence intervals
metrics_list = ["PCS", "Fe1", "H", "latency_ms"]
metric_labels = {
    "PCS": "Path Coherence Score",
    "Fe1": "Edge F1-Score",
    "H": "Hallucination Rate (%)",
    "latency_ms": "Latency (ms/edge)"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_list):
    ax = axes[i]
    stats = statistics["metrics"][metric]
    ci95 = stats["confidence_interval_95"]
    
    x = [0]
    means = [stats["mean"]]
    yerr = [
        [stats["mean"] - ci95["lower"]], 
        [ci95["upper"] - stats["mean"]]
    ]
    
    bar = ax.bar(x, means, yerr=yerr, capsize=8, color='#4a90e2', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value label
    ax.text(
        0, stats["mean"] + (yerr[1][0] * 0.1),
        f"{stats['mean']:.4f}",
        ha='center', va='bottom', fontweight='bold', fontsize=11
    )
    
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(metric_labels[metric])
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add CI text
    ax.text(
        0.5, 0.05,
        f"95% CI: [{ci95['lower']:.3f}, {ci95['upper']:.3f}]",
        ha='center', va='bottom', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

plt.tight_layout()
plt.savefig(output_dir / "demo_metrics_barplot.pdf", dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'demo_metrics_barplot.pdf'}")

# 2. Box plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_list):
    ax = axes[i]
    values = [r[metric] for r in results]
    bp = ax.boxplot(values, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#50c878')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(f"Distribution of {metric_labels[metric]} (n=25)")
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "demo_metrics_boxplot.pdf", dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'demo_metrics_boxplot.pdf'}")

# 3. Correlation between PCS and H
pcs_values = np.array([r["PCS"] for r in results])
h_values = np.array([r["H"] for r in results])
corr = np.corrcoef(pcs_values, h_values)[0, 1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(pcs_values, h_values, alpha=0.7, s=80, color='#4a90e2', edgecolors='black', linewidth=0.8)

# Add trend line
z = np.polyfit(pcs_values, h_values, 1)
p = np.poly1d(z)
ax.plot(pcs_values, p(pcs_values), "r--", alpha=0.8, linewidth=2, label=f"Trend line (r={corr:.3f})")

ax.set_xlabel("Path Coherence Score (PCS)")
ax.set_ylabel("Hallucination Rate (%)")
ax.set_title("Correlation between PCS and Hallucination Rate")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "demo_pcs_h_correlation.pdf", dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir / 'demo_pcs_h_correlation.pdf'}")

print("\nDone! All plots saved to demo_results/plots/")
print("\nSummary of results:")
for metric in metrics_list:
    s = statistics["metrics"][metric]
    print(f"  {metric_labels[metric]:25} : {s['mean']:.4f} ± {s['std']:.4f}")
