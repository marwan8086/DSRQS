# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Publication-quality visualization tools for scientific papers
#   - Bar plots
#   - Box plots
#   - Training curves
#   - Result comparison plots
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.format'] = 'pdf'


def plot_training_curves(
    history_list: List[List[Dict]],
    output_path: Path,
    title: str = "Training Curves",
    metrics: Optional[List[str]] = None
):
    """
    Plot training curves (loss and validation metrics).
    
    Args:
        history_list: List of training histories (one per run)
        output_path: Path to save figure
        title: Plot title
        metrics: Metrics to plot (defaults to train_loss, PCS)
    """
    set_publication_style()
    
    if metrics is None:
        metrics = ["train_loss", "PCS"]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_list)))
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for run_idx, history in enumerate(history_list):
            epochs = [h["epoch"] for h in history]
            
            if metric == "train_loss":
                values = [h["train_loss"] for h in history]
            else:
                values = [h["val_metrics"][metric] for h in history]
            
            ax.plot(epochs, values, color=colors[run_idx], alpha=0.3, linewidth=1)
        
        # Plot mean and std
        all_values = defaultdict(list)
        for history in history_list:
            for h in history:
                if metric == "train_loss":
                    val = h["train_loss"]
                else:
                    val = h["val_metrics"][metric]
                all_values[h["epoch"]].append(val)
        
        epochs_sorted = sorted(all_values.keys())
        means = [np.mean(all_values[e]) for e in epochs_sorted]
        stds = [np.std(all_values[e]) for e in epochs_sorted]
        
        ax.plot(epochs_sorted, means, color='black', linewidth=2, label='Mean')
        ax.fill_between(epochs_sorted, 
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       color='gray', alpha=0.2, label='±1 Std')
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        if metric_idx == num_metrics - 1:
            ax.legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_result_comparison(
    results_dict: Dict[str, List[Dict]],
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_confidence: bool = True,
    rotate_labels: bool = True
):
    """
    Plot bar chart comparing different variants.
    
    Args:
        results_dict: Dictionary mapping variant names to list of results
        metric: Metric to plot
        output_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        show_confidence: Whether to show confidence intervals
        rotate_labels: Whether to rotate x-axis labels
    """
    set_publication_style()
    
    variants = list(results_dict.keys())
    means = []
    cis_lower = []
    cis_upper = []
    
    for variant in variants:
        data = np.array([r[metric] for r in results_dict[variant]])
        means.append(np.mean(data))
        
        if show_confidence:
            from .statistics import calculate_confidence_interval
            ci = calculate_confidence_interval(data)
            cis_lower.append(ci["lower"])
            cis_upper.append(ci["upper"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(variants))
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
    
    bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    if show_confidence:
        ax.errorbar(x_pos, means,
                   yerr=[np.array(means) - np.array(cis_lower), np.array(cis_upper) - np.array(means)],
                   fmt='none', ecolor='black', capsize=5, linewidth=1.5)
    
    # Add value labels on top of bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Variant")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.set_xticks(x_pos)
    
    if rotate_labels:
        plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot_comparison(
    results_dict: Dict[str, List[Dict]],
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotate_labels: bool = True
):
    """
    Plot boxplot comparing different variants.
    
    Args:
        results_dict: Dictionary mapping variant names to list of results
        metric: Metric to plot
        output_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        rotate_labels: Whether to rotate x-axis labels
    """
    set_publication_style()
    
    variants = list(results_dict.keys())
    data_list = []
    
    for variant in variants:
        data_list.append(np.array([r[metric] for r in results_dict[variant]]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(data_list, patch_artist=True)
    
    # Customize colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Customize other elements
    for element in ['whiskers', 'caps', 'medians']:
        for e in bp[element]:
            e.set_color('black')
            e.set_linewidth(1.5)
    
    ax.set_xlabel("Variant")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.set_xticklabels(variants)
    
    if rotate_labels:
        plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_correlation(
    results: List[Dict],
    metric1: str,
    metric2: str,
    output_path: Path,
    title: Optional[str] = None
):
    """
    Plot scatter plot showing correlation between two metrics.
    
    Args:
        results: List of result dictionaries
        metric1: First metric
        metric2: Second metric
        output_path: Path to save figure
        title: Plot title
    """
    set_publication_style()
    
    x = np.array([r[metric1] for r in results])
    y = np.array([r[metric2] for r in results])
    
    # Calculate correlation
    corr = np.corrcoef(x, y)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend (R={corr:.3f})')
    
    ax.set_xlabel(metric1)
    ax.set_ylabel(metric2)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_visualizations(
    results_dir: Path,
    output_dir: Path
):
    """
    Generate all standard visualizations from result files.
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    results_dict = {}
    history_dict = {}
    
    for result_file in results_dir.rglob("results.json"):
        variant_name = result_file.parent.name
        with open(result_file, 'r') as f:
            data = json.load(f)
            results_dict[variant_name] = data["results"]
    
    # Load histories if available
    for history_file in results_dir.rglob("history_*.json"):
        parts = history_file.stem.split("_")
        seed = int(parts[1])
        fold = int(parts[2])
        variant_name = history_file.parent.name
        
        if variant_name not in history_dict:
            history_dict[variant_name] = []
        
        with open(history_file, 'r') as f:
            history_dict[variant_name].append(json.load(f))
    
    # Generate plots
    metrics = ["PCS", "Fe1", "H", "latency_ms"]
    
    for metric in metrics:
        plot_result_comparison(
            results_dict,
            metric,
            output_dir / f"comparison_{metric}.pdf",
            title=f"Comparison of {metric}"
        )
        
        plot_boxplot_comparison(
            results_dict,
            metric,
            output_dir / f"boxplot_{metric}.pdf",
            title=f"Boxplot Comparison of {metric}"
        )
    
    # Plot training curves for each variant
    for variant_name, histories in history_dict.items():
        plot_training_curves(
            histories,
            output_dir / f"training_{variant_name}.pdf",
            title=f"Training Curves - {variant_name}"
        )
    
    # Plot correlation between PCS and H (hallucination rate)
    if "dsrqs" in results_dict:
        plot_metric_correlation(
            results_dict["dsrqs"],
            "PCS",
            "H",
            output_dir / "correlation_pcs_h.pdf",
            title="Correlation between PCS and Hallucination Rate"
        )
