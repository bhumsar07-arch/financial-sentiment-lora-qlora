"""
Visualization Module for Financial Sentiment PEFT Research.

Generates publication-quality charts saved to results/figures/:
1. Class distribution & split breakdown
2. Sentence length distributions
3. Confusion matrices
4. Method comparisons (Zero-shot vs LoRA vs QLoRA)
5. Efficiency comparisons (VRAM, Training Time, Trainable %)
6. Rank ablation curves (Rank vs F1, Params, Memory, Time, Adapter Size)
7. Performance vs Efficiency trade-off scatter plots
"""

import os
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Set consistent style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


def ensure_fig_dir(output_dir: str = "results/figures") -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label_text",
    save_path: str = "results/figures/class_distribution.png",
) -> None:
    """Plot overall sentiment class distribution."""
    ensure_fig_dir(os.path.dirname(save_path))
    fig, ax = plt.subplots(figsize=(7, 5))

    counts = df[label_col].value_counts()
    colors = ["#4a90e2", "#7ed321", "#d0021b"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor="black", alpha=0.85)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(counts.values) * 0.015,
            f"{int(yval)} ({yval/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Financial PhraseBank Class Distribution", pad=15)
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label_text",
    save_path: str = "results/figures/split_distribution.png",
) -> None:
    """Plot stratified class breakdown across train/val/test splits."""
    ensure_fig_dir(os.path.dirname(save_path))

    train_c = train_df[label_col].value_counts()
    val_c = val_df[label_col].value_counts()
    test_c = test_df[label_col].value_counts()

    combined_df = pd.DataFrame({
        "Train (70%)": train_c,
        "Validation (15%)": val_c,
        "Test (15%)": test_c,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    combined_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.75, edgecolor="black", alpha=0.9)

    ax.set_title("Stratified Dataset Splits (70 / 15 / 15)", pad=15)
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Sample Count")
    ax.legend(title="Split")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sentence_lengths(
    df: pd.DataFrame,
    text_col: str = "sentence",
    save_path: str = "results/figures/sentence_length_distribution.png",
) -> None:
    """Plot distribution of sentence token/word counts."""
    ensure_fig_dir(os.path.dirname(save_path))
    lengths = df[text_col].astype(str).apply(lambda s: len(s.split()))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths, kde=True, color="#2b5c8f", bins=30, ax=ax, edgecolor="black")

    median_len = lengths.median()
    mean_len = lengths.mean()
    p95_len = np.percentile(lengths, 95)

    ax.axvline(median_len, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_len:.0f}")
    ax.axvline(p95_len, color="darkorange", linestyle=":", linewidth=1.5, label=f"95th Pct: {p95_len:.0f}")

    ax.set_title("Financial PhraseBank Sentence Length Distribution", pad=15)
    ax.set_xlabel("Sentence Length (Words)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm_dict: Dict[str, Any],
    method_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot an annotated confusion matrix heatmap."""
    labels = cm_dict["labels"]
    matrix_2d = np.array(cm_dict["matrix_2d"])

    if save_path is None:
        save_path = f"results/figures/cm_{method_name.lower().replace(' ', '_')}.png"
    ensure_fig_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix_2d,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor="white",
    )

    ax.set_title(f"Confusion Matrix: {method_name}", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_comparison(
    results_map: Dict[str, Dict[str, Any]],
    save_path: str = "results/figures/performance_comparison.png",
) -> None:
    """
    Compare Accuracy, Macro-F1, and Weighted-F1 across evaluated methods.
    Only includes methods with status != 'NOT_RUN'.
    """
    ensure_fig_dir(os.path.dirname(save_path))

    methods, accs, f1s, weighted_f1s = [], [], [], []
    for method, data in results_map.items():
        if data and data.get("status") != "NOT_RUN" and "macro_f1" in data:
            methods.append(method)
            accs.append(data.get("accuracy", 0.0))
            f1s.append(data.get("macro_f1", 0.0))
            weighted_f1s.append(data.get("weighted_f1", 0.0))

    if not methods:
        return

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    r1 = ax.bar(x - width, accs, width, label="Accuracy", color="#2b5c8f", edgecolor="black")
    r2 = ax.bar(x, f1s, width, label="Macro F1", color="#e7298a", edgecolor="black")
    r3 = ax.bar(x + width, weighted_f1s, width, label="Weighted F1", color="#66a61e", edgecolor="black")

    ax.set_title("Predictive Performance Comparison", pad=15)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bars in [r1, r2, r3]:
        for b in bars:
            val = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_efficiency_comparison(
    results_map: Dict[str, Dict[str, Any]],
    save_path: str = "results/figures/efficiency_comparison.png",
) -> None:
    """Compare Peak VRAM (GB) and Training Time across evaluated methods."""
    ensure_fig_dir(os.path.dirname(save_path))

    methods, vram_list, time_list = [], [], []
    for method, data in results_map.items():
        if data and data.get("status") != "NOT_RUN":
            methods.append(method)
            vram_list.append(data.get("peak_vram_gb", data.get("peak_memory_gb", 0.0)))
            time_list.append(data.get("training_time_seconds", 0.0) / 60.0)

    if not methods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # VRAM Plot
    ax1.bar(methods, vram_list, color=["#1b9e77", "#d95f02", "#7570b3"][:len(methods)], edgecolor="black", width=0.55)
    ax1.set_title("Peak GPU Memory (VRAM)", pad=15)
    ax1.set_ylabel("Peak Memory (GB)")
    for i, v in enumerate(vram_list):
        ax1.text(i, v + 0.3, f"{v:.1f} GB", ha="center", fontweight="bold")
    ax1.set_ylim(0, max(vram_list) * 1.2 if vram_list and max(vram_list) > 0 else 10)

    # Training Time Plot (exclude Zero-shot if zero)
    train_methods = [m for m, t in zip(methods, time_list) if t > 0]
    train_times = [t for t in time_list if t > 0]
    if train_methods:
        ax2.bar(train_methods, train_times, color=["#d95f02", "#7570b3"][:len(train_methods)], edgecolor="black", width=0.45)
        ax2.set_title("Training Duration", pad=15)
        ax2.set_ylabel("Time (Minutes)")
        for i, t in enumerate(train_times):
            ax2.text(i, t + 0.2, f"{t:.1f} min", ha="center", fontweight="bold")
        ax2.set_ylim(0, max(train_times) * 1.2 if train_times else 10)
    else:
        ax2.text(0.5, 0.5, "No training run yet", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rank_ablation(
    ablation_results: List[Dict[str, Any]],
    save_path: str = "results/figures/rank_ablation.png",
) -> None:
    """Plot LoRA Rank vs F1, Trainable Params, Memory, and Training Time."""
    ensure_fig_dir(os.path.dirname(save_path))
    if not ablation_results:
        return

    ranks = [r["rank"] for r in ablation_results]
    f1s = [r.get("macro_f1", 0.0) for r in ablation_results]
    trainable_params_m = [r.get("trainable_parameters_millions", 0.0) for r in ablation_results]
    vram = [r.get("peak_vram_gb", 0.0) for r in ablation_results]
    training_times = [r.get("training_time_seconds", 0.0) / 60.0 for r in ablation_results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # Rank vs Macro F1
    ax1.plot(ranks, f1s, marker="o", color="#e7298a", linewidth=2.5, markersize=8)
    ax1.set_title("LoRA Rank vs. Macro F1", pad=10)
    ax1.set_xlabel("Rank (r)")
    ax1.set_ylabel("Macro F1")
    ax1.set_xticks(ranks)

    # Rank vs Trainable Parameters
    ax2.plot(ranks, trainable_params_m, marker="s", color="#2b5c8f", linewidth=2.5, markersize=8)
    ax2.set_title("LoRA Rank vs. Trainable Parameters (Millions)", pad=10)
    ax2.set_xlabel("Rank (r)")
    ax2.set_ylabel("Parameters (M)")
    ax2.set_xticks(ranks)

    # Rank vs Peak VRAM
    ax3.plot(ranks, vram, marker="^", color="#1b9e77", linewidth=2.5, markersize=8)
    ax3.set_title("LoRA Rank vs. Peak GPU VRAM", pad=10)
    ax3.set_xlabel("Rank (r)")
    ax3.set_ylabel("Peak VRAM (GB)")
    ax3.set_xticks(ranks)

    # Rank vs Training Time
    ax4.plot(ranks, training_times, marker="d", color="#d95f02", linewidth=2.5, markersize=8)
    ax4.set_title("LoRA Rank vs. Training Time", pad=10)
    ax4.set_xlabel("Rank (r)")
    ax4.set_ylabel("Duration (Minutes)")
    ax4.set_xticks(ranks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_vs_efficiency(
    results_map: Dict[str, Dict[str, Any]],
    save_path: str = "results/figures/performance_vs_cost_tradeoff.png",
) -> None:
    """Plot 2D trade-off: Model Quality (Macro-F1) vs Computational Cost (Peak VRAM)."""
    ensure_fig_dir(os.path.dirname(save_path))

    methods, f1s, vrams = [], [], []
    for method, data in results_map.items():
        if data and data.get("status") != "NOT_RUN" and "macro_f1" in data:
            methods.append(method)
            f1s.append(data.get("macro_f1", 0.0))
            vrams.append(data.get("peak_vram_gb", data.get("peak_memory_gb", 0.0)))

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"][:len(methods)]

    for i, method in enumerate(methods):
        ax.scatter(vrams[i], f1s[i], s=250, color=colors[i], edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(
            method,
            (vrams[i], f1s[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Model Quality vs. GPU Resource Cost Trade-Off", pad=15)
    ax.set_xlabel("Peak GPU Memory (GB)")
    ax.set_ylabel("Predictive Performance (Macro F1)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
