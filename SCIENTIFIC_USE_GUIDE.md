# DSRQS — A Beginner-Friendly Guide

This guide walks you through using DSRQS, step by step, in plain language. You don't
need to be an expert in machine learning or statistics to follow along — anything
technical is explained the first time it comes up.

## What is DSRQS, in simple terms?

DSRQS is a research method for working with **biomedical knowledge graphs** — large
networks that connect medical concepts (like diseases, genes, and symptoms) with the
relationships between them. The goal is to keep the *correct* connections and filter
out the *wrong* ones.

This repository gives you two things:

1. **The method itself** (DSRQS), and
2. **A toolkit** that runs experiments, measures how well the method performs, and
   produces clean charts and tables you can put straight into a scientific paper.

Everything is designed so your results can be **reproduced** — meaning anyone who runs
the same commands gets the same numbers. That's a core requirement for trustworthy
science.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running Your First Experiment](#running-your-first-experiment)
3. [Understanding the Results](#understanding-the-results)
4. [The Charts It Makes](#the-charts-it-makes)
5. [A Full Walkthrough](#a-full-walkthrough)
6. [What the Numbers Mean](#what-the-numbers-mean)
7. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Getting Started

Before running anything, install the tools the project depends on. Open a terminal in
the project folder and run:

```bash
pip install -r requirements.txt
```

This reads a list of required software packages and installs them for you. You only
need to do this once.

---

## Running Your First Experiment

There are three things you'll typically do, in order.

### Step 1 — Prepare the data

```bash
python main.py --dataset orphanet_fq274 --mode prepare_data
```

This splits your dataset into groups so the method can be tested fairly. It uses
something called **5-fold cross-validation**: the data is divided into 5 equal parts,
and the method is tested 5 times — each time using a different part for testing and the
rest for training. Averaging across all 5 gives a more reliable score than testing just
once. The split is saved to `data/split_info.json` so the exact same division can be
reused later.

### Step 2 — Run the experiment

To run a single method and fully evaluate it:

```bash
python main.py --dataset orphanet_fq274 --variant dsrqs --mode full_eval --generate_plots
```

To run *all* the comparison methods at once (this reproduces the complete results from
the paper):

```bash
python main.py --dataset orphanet_fq274 --mode all_baselines --generate_plots
```

Adding `--generate_plots` tells the toolkit to also create the charts. Every run is
saved into its own folder, stamped with the date and time, so nothing ever gets
overwritten.

### Step 3 — Re-analyze old results (optional)

If you already ran experiments before and just want to re-process them:

```bash
python main.py --dataset orphanet_fq274 --mode analyze --results_dir ./results/your_experiment --generate_plots
```

---

## Understanding the Results

After a run, the toolkit doesn't just give you raw scores — it checks whether those
scores are **statistically meaningful**. Here's what that involves, explained plainly.

### Confidence intervals

A confidence interval is a range that tells you how sure you can be about a result.
Saying "the score is 0.85" is less honest than saying "the score is 0.85, and we're 95%
confident the true value sits between 0.82 and 0.88." The toolkit reports these ranges
automatically.

### Statistical tests

When comparing two methods, a small difference might just be luck. Statistical tests
estimate how likely that is. The key output is a **p-value**: a small p-value (commonly
below 0.05) means the difference is probably real, not random chance.

### Effect size

A difference can be real but tiny. **Effect size** measures *how big* the difference is,
not just whether it exists. Together, the p-value and effect size tell you whether one
method is meaningfully better than another.

The toolkit compares every method against a baseline and saves the verdict in a file,
roughly like this:

```json
{
  "baseline": "cosine",
  "metric": "PCS",
  "variants": {
    "dsrqs": {
      "statistical_test": {
        "p_value": 0.001,
        "effect_size_r": 0.8,
        "significant": true
      }
    }
  }
}
```

In plain words: this says DSRQS beat the baseline, the difference is very unlikely to be
chance (`p_value` is tiny), the improvement is large (`effect_size_r` is high), and the
result counts as significant.

---

## The Charts It Makes

When you use `--generate_plots`, the toolkit produces ready-to-publish figures (saved as
high-resolution PDFs):

- **Comparison charts** — bar charts showing each method's score side by side, with the
  confidence ranges drawn on top.
- **Box plots** — a quick visual of how spread out the results were across all the test
  runs.
- **Training curves** — how the method improved over time as it learned.
- **Correlation charts** — whether two things rise and fall together (for example, does a
  higher quality score go hand in hand with fewer mistakes?).

All charts share a clean, consistent style with clear labels — the kind you can drop
directly into a paper.

---

## A Full Walkthrough

Here's the whole process from start to finish:

```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Prepare the data
python main.py --dataset orphanet_fq274 --mode prepare_data

# 3. Run all methods and generate charts + statistics
python main.py --dataset orphanet_fq274 --mode all_baselines --generate_plots
```

When it finishes, you'll get a folder like this:

```
results/
└── orphanet_fq274_all_baselines_20260605_123456/
    ├── experiment_metadata.json    ← exactly how this run was set up
    ├── statistical_comparison.json ← which method won, and by how much
    ├── results_table.tex           ← a table ready to paste into a LaTeX paper
    ├── figures/                    ← all the PDF charts
    │   ├── comparison_PCS.pdf
    │   ├── boxplot_PCS.pdf
    │   └── ...
    └── dsrqs/                      ← detailed results for each method
        ├── results.json
        └── logs/
```

To use it in a paper: paste the table from `results_table.tex`, drop in the PDF charts,
and report the p-values and effect sizes from `statistical_comparison.json`.

---

## What the Numbers Mean

| Metric | What it measures | Which direction is good? |
|--------|------------------|--------------------------|
| **PCS** (Path Coherence Score) | How well the method keeps complete, sensible chains of connections | Higher is better |
| **F1** | How accurately it identifies correct connections | Higher is better |
| **H** (Hallucination Rate) | How often it invents connections that aren't real | Lower is better |
| **Δα** (Depth Awareness) | How well it accounts for how "deep" a connection sits in the graph | Higher is better |
| **Latency** | How long it takes to process each connection | Lower is faster |

A "hallucination," in this context, is when the method confidently produces something
that simply isn't true — a known weakness in many AI systems, which is exactly why
measuring it matters.

---

## Tips & Troubleshooting

- **Run with all 5 seeds** (the default). A "seed" sets the random starting point; using
  several and averaging makes your results far more trustworthy than a single lucky run.
- **Always add `--generate_plots`** when you want publication-ready figures.
- **Let early stopping do its job.** The toolkit automatically stops training once the
  method stops improving, which saves time and avoids over-fitting (memorizing the data
  instead of learning the pattern).
- **Everything is documented for you.** Each run records its full setup, so you can
  always reproduce or explain exactly what you did.

If you get stuck, two other files can help:

- `QUICKSTART.md` — the shortest possible path to running it.
- `IMPLEMENTATION_GUIDE.md` — the deep technical details for developers.
