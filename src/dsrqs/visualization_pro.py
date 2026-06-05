# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Visualization library for DSRQS results, providing publication-quality
#   scientific figures (bar, line, ROC/PR, confusion matrix, radar, etc.).
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

from typing import List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix, auc
)

plt.rcParams.update({
    "figure.dpi": 300,
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True,
    "font.family": "serif",
})


class PlotType(Enum):
    """Enumeration of all available plot types."""
    BAR = auto()
    LINE = auto()
    SCATTER = auto()
    HEATMAP = auto()
    ROC = auto()
    PR = auto()
    CONFUSION_MATRIX = auto()
    RADAR = auto()
    BOX = auto()
    VIOLIN = auto()
    HISTOGRAM = auto()
    PIE = auto()
    ERROR_BAR = auto()
    HIST2D = auto()
    QUANTILE = auto()
    STAIRS = auto()
    FILL_BETWEEN = auto()
    STACKED_BAR = auto()
    GROUPED_BAR = auto()
    DOT = auto()
    SWARM = auto()
    STRIP = auto()
    JOINT = auto()
    PAIR = auto()
    HEXBIN = auto()
    KDE = auto()
    ECDF = auto()
    BAYESIAN = auto()
    FOREST = auto()
    SPIDER = auto()
    POLAR = auto()
    DENDROGRAM = auto()
    TREE = auto()
    GRAPH = auto()
    NETWORK = auto()
    TIME_SERIES = auto()
    CUMULATIVE = auto()
    RESIDUAL = auto()
    QQ = auto()
    PP = auto()
    WATERFALL = auto()
    GAUGE = auto()
    DONUT = auto()
    SANKEY = auto()
    CHORD = auto()
    BUBBLE = auto()
    TERNARY = auto()
    # NOTE: identifiers cannot start with a digit, so 3D_* was renamed *_3D.
    SCATTER_3D = auto()
    SURFACE_3D = auto()
    CONTOUR_3D = auto()
    WIRE_3D = auto()
    CORRELATION = auto()
    CLUSTER = auto()
    ELBOW = auto()
    SILHOUETTE = auto()
    LEARNING_CURVE = auto()
    VALIDATION_CURVE = auto()
    PARTIAL_DEPENDENCE = auto()
    SHAP = auto()
    LIME = auto()


@dataclass
class PlotConfig:
    """Configuration for all plot types."""
    title: str = "DSRQS Results"
    xlabel: str = "X Axis"
    ylabel: str = "Y Axis"
    figsize: Tuple[int, int] = (10, 6)
    cmap: str = "viridis"
    color: str = "tab:blue"
    alpha: float = 0.7
    grid: bool = True
    legend: bool = True
    save_path: Optional[str] = None
    show: bool = True
    font_family: str = "serif"
    font_size: int = 12
    dpi: int = 300


def _finalize(config: PlotConfig) -> None:
    """Common save/show/close logic shared by every plot function."""
    if config.save_path:
        plt.savefig(config.save_path, dpi=config.dpi, bbox_inches="tight")
    if config.show:
        plt.show()
    plt.close()


def plot_accuracy_bar(
    accuracies: List[float],
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot accuracy comparison bar chart."""
    config = config or PlotConfig()
    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(accuracies))]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    bars = plt.bar(labels, accuracies, color=config.color, alpha=config.alpha)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.xlabel(config.xlabel, fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{height:.4f}", ha="center", va="bottom")

    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def plot_loss_curve(
    train_loss: List[float],
    val_loss: List[float],
    config: Optional[PlotConfig] = None,
):
    """Plot training and validation loss curves."""
    config = config or PlotConfig()
    plt.figure(figsize=config.figsize, dpi=config.dpi)
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="tab:orange", linewidth=2)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.xlabel("Epoch", fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel("Loss", fontfamily=config.font_family, fontsize=config.font_size)
    plt.grid(True, alpha=0.3)
    if config.legend:
        plt.legend()
    _finalize(config)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot confusion matrix."""
    config = config or PlotConfig()
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = [f"Class {i}" for i in range(cm.shape[0])]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    im = plt.imshow(cm, interpolation="nearest", cmap=config.cmap)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.colorbar(im)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label", fontfamily=config.font_family, fontsize=config.font_size)
    plt.xlabel("Predicted label", fontfamily=config.font_family, fontsize=config.font_size)
    plt.tight_layout()
    _finalize(config)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    config: Optional[PlotConfig] = None,
):
    """Plot ROC curve."""
    config = config or PlotConfig()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    plt.plot(fpr, tpr, color=config.color, lw=2,
             label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel("True Positive Rate", fontfamily=config.font_family, fontsize=config.font_size)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    if config.legend:
        plt.legend(loc="lower right")
    if config.grid:
        plt.grid(True, alpha=0.3)
    _finalize(config)


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    config: Optional[PlotConfig] = None,
):
    """Plot precision-recall curve."""
    config = config or PlotConfig()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    plt.plot(recall, precision, color=config.color, lw=2,
             label=f"PR curve (area = {pr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel("Precision", fontfamily=config.font_family, fontsize=config.font_size)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    if config.legend:
        plt.legend(loc="lower left")
    if config.grid:
        plt.grid(True, alpha=0.3)
    _finalize(config)


def plot_violin(
    data: List[np.ndarray],
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot violin plot for multiple distributions."""
    config = config or PlotConfig()
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(data))]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    parts = plt.violinplot(data, showmeans=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor(config.color)
        pc.set_alpha(config.alpha)

    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def plot_heatmap(
    data: np.ndarray,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot heatmap from 2D data."""
    config = config or PlotConfig()
    if xticklabels is None:
        xticklabels = [str(i) for i in range(data.shape[1])]
    if yticklabels is None:
        yticklabels = [str(i) for i in range(data.shape[0])]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    im = plt.imshow(data, cmap=config.cmap, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(xticklabels)), xticklabels, rotation=45, ha="right")
    plt.yticks(range(len(yticklabels)), yticklabels)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.xlabel(config.xlabel, fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    _finalize(config)


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[np.ndarray] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot scatter plot, optionally colored by class label."""
    config = config or PlotConfig()
    plt.figure(figsize=config.figsize, dpi=config.dpi)

    if labels is not None:
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(x[mask], y[mask], alpha=config.alpha, label=str(label))
        if config.legend:
            plt.legend()
    else:
        plt.scatter(x, y, color=config.color, alpha=config.alpha)

    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.xlabel(config.xlabel, fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    if config.grid:
        plt.grid(True, alpha=0.3)
    _finalize(config)


def plot_boxplot(
    data: List[np.ndarray],
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot box plot for multiple distributions."""
    config = config or PlotConfig()
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(data))]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    # NOTE: the `labels=` kwarg is deprecated in recent matplotlib; set ticks
    # manually so this works across versions.
    bp = plt.boxplot(data, patch_artist=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)

    for patch in bp["boxes"]:
        patch.set_facecolor(config.color)
        patch.set_alpha(config.alpha)

    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def plot_histogram(
    data: np.ndarray,
    bins: int = 20,
    config: Optional[PlotConfig] = None,
):
    """Plot histogram."""
    config = config or PlotConfig()
    plt.figure(figsize=config.figsize, dpi=config.dpi)
    plt.hist(data, bins=bins, color=config.color, alpha=config.alpha, edgecolor="black")
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.xlabel(config.xlabel, fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylabel(config.ylabel, fontfamily=config.font_family, fontsize=config.font_size)
    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def plot_radar(
    categories: List[str],
    values: List[float],
    config: Optional[PlotConfig] = None,
):
    """Plot radar chart. Input lists are not mutated."""
    config = config or PlotConfig()
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the loop on COPIES so the caller's lists stay untouched.
    closed_values = list(values) + [values[0]]
    closed_angles = angles + [angles[0]]

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    ax = plt.subplot(111, polar=True)
    ax.plot(closed_angles, closed_values, color=config.color, linewidth=2, linestyle="solid")
    ax.fill(closed_angles, closed_values, color=config.color, alpha=config.alpha)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    _finalize(config)


def plot_correlation_matrix(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """Plot correlation matrix heatmap."""
    config = config or PlotConfig()
    corr = np.corrcoef(data.T)
    if labels is None:
        labels = [f"Feature {i + 1}" for i in range(corr.shape[0])]
    plot_heatmap(corr, xticklabels=labels, yticklabels=labels, config=config)


def plot_depth_performance(
    depths: List[int],
    tprs: List[float],
    fprs: List[float],
    config: Optional[PlotConfig] = None,
):
    """Plot depth-specific true/false positive rates."""
    config = config or PlotConfig()
    x = np.arange(len(depths))
    width = 0.35

    plt.figure(figsize=config.figsize, dpi=config.dpi)
    plt.bar(x - width / 2, tprs, width, label="TPR", color="tab:green")
    plt.bar(x + width / 2, fprs, width, label="FPR", color="tab:red")

    plt.xticks(x, [f"Depth {d}" for d in depths])
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.ylabel("Rate", fontfamily=config.font_family, fontsize=config.font_size)
    plt.legend()
    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def plot_pcs_comparison(
    methods: List[str],
    pcs_values: List[float],
    config: Optional[PlotConfig] = None,
):
    """Plot Path-Coherence Score (PCS) comparison bar chart."""
    config = config or PlotConfig()
    plt.figure(figsize=config.figsize, dpi=config.dpi)
    bars = plt.bar(methods, pcs_values, color="tab:blue", alpha=0.8)
    plt.title(config.title, fontfamily=config.font_family, fontsize=config.font_size + 2)
    plt.ylabel("Path-Coherence Score (PCS)", fontfamily=config.font_family, fontsize=config.font_size)
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{height:.3f}", ha="center", va="bottom")

    plt.xticks(rotation=45, ha="right")
    if config.grid:
        plt.grid(axis="y", alpha=0.3)
    _finalize(config)


def demo_all_plots(show: bool = False, outdir: str = "."):
    """Run a demo of all plot types, saving figures to `outdir`."""
    import os
    os.makedirs(outdir, exist_ok=True)

    def cfg(title: str, fname: str) -> PlotConfig:
        return PlotConfig(title=title, save_path=os.path.join(outdir, fname), show=show)

    print("=" * 80)
    print("VISUALIZATION DEMO - Generating all plot types")
    print("=" * 80)

    rng = np.random.default_rng(42)

    print("\n1. Accuracy bar plot...")
    accs = rng.uniform(0.7, 0.95, 5).tolist()
    plot_accuracy_bar(accs, ["A", "B", "C", "D", "E"], cfg("Accuracy Comparison", "demo_accuracy_bar.png"))

    print("2. Loss curves...")
    train_loss = (np.linspace(1.0, 0.1, 50) + rng.normal(0, 0.05, 50)).tolist()
    val_loss = (np.linspace(1.0, 0.15, 50) + rng.normal(0, 0.07, 50)).tolist()
    plot_loss_curve(train_loss, val_loss, cfg("Loss Curves", "demo_loss_curves.png"))

    print("3. Confusion matrix...")
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    y_pred[:70] = y_true[:70]
    plot_confusion_matrix(y_true, y_pred, ["Negative", "Positive"], cfg("Confusion Matrix", "demo_confusion_matrix.png"))

    print("4. ROC curve...")
    y_score = y_true * 0.7 + rng.random(100) * 0.3
    plot_roc_curve(y_true, y_score, cfg("ROC Curve", "demo_roc.png"))

    print("5. PR curve...")
    plot_pr_curve(y_true, y_score, cfg("Precision-Recall Curve", "demo_pr.png"))

    print("6. Violin plot...")
    d1 = rng.normal(0, 1, 100)
    d2 = rng.normal(1, 1.5, 100)
    d3 = rng.normal(-1, 0.5, 100)
    plot_violin([d1, d2, d3], ["X", "Y", "Z"], cfg("Violin Plot", "demo_violin.png"))

    print("7. Scatter plot...")
    x = rng.standard_normal(100)
    y = x * 0.8 + rng.standard_normal(100) * 0.3
    plot_scatter(x, y, config=cfg("Scatter Plot", "demo_scatter.png"))

    print("8. Heatmap...")
    plot_heatmap(rng.random((5, 5)), config=cfg("Heatmap", "demo_heatmap.png"))

    print("9. Box plot...")
    plot_boxplot([d1, d2, d3], ["X", "Y", "Z"], cfg("Box Plot", "demo_box.png"))

    print("10. Histogram...")
    plot_histogram(x, config=cfg("Histogram", "demo_histogram.png"))

    print("11. Radar plot...")
    plot_radar(["A", "B", "C", "D", "E"], rng.random(5).tolist(), cfg("Radar Chart", "demo_radar.png"))

    print("12. Correlation matrix...")
    plot_correlation_matrix(rng.standard_normal((100, 5)), config=cfg("Correlation Matrix", "demo_correlation.png"))

    print("13. Depth performance...")
    plot_depth_performance([1, 2, 3], [0.9, 0.85, 0.8], [0.1, 0.12, 0.15],
                           cfg("Depth-Specific Performance", "demo_depth_performance.png"))

    print("14. PCS comparison...")
    plot_pcs_comparison(["Baseline", "Depth-Aware", "DSRQS"], [0.6, 0.75, 0.85],
                        cfg("PCS Comparison", "demo_pcs_comparison.png"))

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_all_plots(show=False, outdir="demo_figures")