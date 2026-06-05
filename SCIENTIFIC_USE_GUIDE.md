# DSRQS Scientific Use Guide

This guide explains how to use the DSRQS framework for conducting rigorous scientific experiments that meet high academic standards.

## Table of Contents

1. [Reproducibility](#reproducibility)
2. [Statistical Analysis](#statistical-analysis)
3. [Visualization](#visualization)
4. [Full Workflow Example](#full-workflow-example)

---

## Reproducibility

### 1. Preparing Data Splits

First, prepare reproducible data splits:

```bash
python main.py --dataset orphanet_fq274 --mode prepare_data
```

This generates:
- 5-fold cross-validation splits for all datasets
- `data/split_info.json` file with full split documentation

### 2. Running Experiments

Run complete experiments:

```bash
# Single variant with full evaluation
python main.py --dataset orphanet_fq274 --variant dsrqs --mode full_eval --generate_plots

# All baselines (complete paper results)
python main.py --dataset orphanet_fq274 --mode all_baselines --generate_plots
```

Each experiment saves:
- Timestamped output directory
- `experiment_metadata.json`: Full environment information
- Training histories per fold
- Checkpoints
- Statistical analysis
- Publication-quality plots (if `--generate_plots` is used)

### 3. Analyzing Existing Results

If you already have results from previous runs:

```bash
python main.py --dataset orphanet_fq274 --mode analyze --results_dir ./results/your_experiment --generate_plots
```

---

## Statistical Analysis

The framework includes comprehensive statistical analysis tools:

### 1. Confidence Intervals
- 95% and 99% confidence intervals using t-distribution
- Bootstrap confidence intervals (10,000 iterations)

### 2. Statistical Tests
- Wilcoxon signed-rank test for paired comparisons
- Mann-Whitney U test for independent samples
- Effect size calculation (Cohen's d)

### 3. Comparative Analysis
Automatically compares all variants against a baseline (default: cosine):

```json
{
  "baseline": "cosine",
  "metric": "PCS",
  "variants": {
    "dsrqs": {
      "baseline_stats": {...},
      "variant_stats": {...},
      "statistical_test": {
        "p_value": 0.001,
        "effect_size_r": 0.8,
        "significant": true
      }
    }
  }
}
```

---

## Visualization

### 1. Publication-Quality Plots

The framework generates the following plots in PDF format (high resolution):

- **Comparison Plots**: Bar charts with confidence intervals for each metric
- **Box Plots**: Distribution of results across all folds/seeds
- **Training Curves**: Loss and metrics over epochs with mean ± std
- **Correlation Plots**: Relationship between PCS and hallucination rate (H)

All plots use:
- Professional font (Arial)
- Consistent styling
- Appropriate color schemes
- Clear labels and legends

---

## Full Workflow Example

Here's a complete workflow from start to finish for a scientific paper:

### Step 1: Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data
```bash
python main.py --dataset orphanet_fq274 --mode prepare_data
```

### Step 3: Run All Experiments
```bash
# This will run all variants and generate all statistics and plots
python main.py --dataset orphanet_fq274 --mode all_baselines --generate_plots
```

### Step 4: Examine Results
```
results/
└── orphanet_fq274_all_baselines_20260605_123456/
    ├── experiment_metadata.json
    ├── statistical_comparison.json
    ├── results_table.tex
    ├── figures/
    │   ├── comparison_PCS.pdf
    │   ├── comparison_F1.pdf
    │   ├── boxplot_PCS.pdf
    │   └── ...
    ├── dsrqs/
    │   ├── results.json
    │   └── logs/
    ├── b6/
    └── ...
```

### Step 5: Use in Paper
1. Copy the LaTeX table template and fill in your results
2. Include the generated PDF figures directly in your paper
3. Report the p-values and effect sizes from the statistical comparison

---

## Metrics Reported

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| PCS | Path Coherence Score | Higher = better (complete paths preserved) |
| F1 | Edge F1-Score | Higher = better edge classification |
| H | Hallucination Rate | Lower = fewer hallucinations |
| Δα | Depth Awareness | Higher = better depth-aware scoring |
| Latency | Inference time per edge | Lower = faster |

---

## Tips for High-Quality Experiments

1. **Use All Seeds**: Always run with 5 seeds (the default)
2. **Generate Plots**: Use `--generate_plots` for publication-quality figures
3. **Check Early Stopping**: The framework uses early stopping by default
4. **Document Everything**: All metadata is automatically saved for reproducibility

---

## Troubleshooting

For issues and questions, refer to:
- `QUICKSTART.md`: Basic usage
- `IMPLEMENTATION_GUIDE.md`: Detailed implementation
