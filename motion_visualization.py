"""
motion_visualization.py

Publication-quality visualization for motion/accelerometer analysis.

Provides:
  - Distribution comparisons (histograms, KDE, box plots, violin plots)
  - Cross-dataset scatter plots and effect size visualizations
  - Cumulative distribution functions
  - Heatmaps of mean predictions across models and classes
  - Frequency analysis plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Set style
plt.style.use('default')

# ─────────────────────────────────────────────────────────────────────────────
# Color Scheme
# ─────────────────────────────────────────────────────────────────────────────

COLOR_FP = '#E74C3C'  # Red
COLOR_TP = '#3498DB'  # Blue
COLOR_NEUTRAL = '#95A5A6'  # Gray

# ─────────────────────────────────────────────────────────────────────────────
# Overall Distribution Comparisons
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_distribution_comparison(fp_features, tp_features, output_dir):
    """
    Create overlaid histograms comparing FP vs TP motion distributions.

    Args:
        fp_features (pd.DataFrame): FP motion features
        tp_features (pd.DataFrame): TP motion features
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['mean_motion', 'max_motion', 'std_motion', 'median_motion']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motion Distribution Comparison: FP vs TP', fontsize=16, fontweight='bold')

    for ax, metric in zip(axes.flat, metrics):
        fp_data = fp_features[metric].dropna()
        tp_data = tp_features[metric].dropna()

        if len(fp_data) == 0 or len(tp_data) == 0:
            continue

        # Histograms with transparency
        ax.hist(fp_data, bins=50, alpha=0.6, label=f'FP (n={len(fp_data)})',
                color=COLOR_FP, edgecolor='black', density=True)
        ax.hist(tp_data, bins=50, alpha=0.6, label=f'TP (n={len(tp_data)})',
                color=COLOR_TP, edgecolor='black', density=True)

        # KDE curves
        fp_kde = stats.gaussian_kde(fp_data)
        tp_kde = stats.gaussian_kde(tp_data)
        x_range = np.linspace(min(fp_data.min(), tp_data.min()),
                              max(fp_data.max(), tp_data.max()), 200)
        ax.plot(x_range, fp_kde(x_range), color=COLOR_FP, linewidth=2, linestyle='--')
        ax.plot(x_range, tp_kde(x_range), color=COLOR_TP, linewidth=2, linestyle='--')

        # Mean lines
        ax.axvline(fp_data.mean(), color=COLOR_FP, linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(tp_data.mean(), color=COLOR_TP, linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Motion (mG)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'distribution_comparison.png')}")
    plt.close()


def plot_box_and_violin_comparison(fp_features, tp_features, output_dir):
    """
    Create side-by-side box and violin plots.

    Args:
        fp_features (pd.DataFrame): FP motion features
        tp_features (pd.DataFrame): TP motion features
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['mean_motion', 'max_motion', 'std_motion', 'median_motion']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motion Distribution (Box & Violin Plots): FP vs TP', fontsize=16, fontweight='bold')

    for ax, metric in zip(axes.flat, metrics):
        fp_data = fp_features[metric].dropna()
        tp_data = tp_features[metric].dropna()

        if len(fp_data) == 0 or len(tp_data) == 0:
            continue

        # Prepare data for plotting
        data_to_plot = [fp_data, tp_data]
        labels = [f'FP\n(n={len(fp_data)})', f'TP\n(n={len(tp_data)})']
        colors = [COLOR_FP, COLOR_TP]

        # Box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Motion (mG)', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'boxplots_comparison.png')}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Dataset Analysis Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_fp_vs_tp(comparison_result, output_dir):
    """
    Scatter plot: FP mean motion vs TP mean motion (per event type).

    Args:
        comparison_result (dict): Comparison results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    per_event = comparison_result['per_event_comparison']
    scatter_data = []

    for event_type, metrics in per_event.items():
        if isinstance(metrics, dict) and 'mean_motion' in metrics:
            comp = metrics['mean_motion']
            scatter_data.append({
                'event_type': event_type,
                'fp_mean': comp['fp_mean'],
                'tp_mean': comp['tp_mean'],
                'ratio': comp['ratio_mean']
            })

    if not scatter_data:
        print("⚠ No data for scatter plot")
        return

    df_scatter = pd.DataFrame(scatter_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(df_scatter['tp_mean'], df_scatter['fp_mean'],
                        s=df_scatter['ratio']*50+50, alpha=0.6,
                        c=df_scatter['ratio'], cmap='RdYlBu_r', edgecolors='black', linewidth=1.5)

    # Add diagonal reference line (1:1)
    max_val = max(df_scatter['fp_mean'].max(), df_scatter['tp_mean'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='1:1 Reference')

    # Labels
    for idx, row in df_scatter.iterrows():
        ax.annotate(row['event_type'], (row['tp_mean'], row['fp_mean']),
                   fontsize=9, ha='right', va='bottom')

    ax.set_xlabel('TP Mean Motion (mG)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FP Mean Motion (mG)', fontsize=12, fontweight='bold')
    ax.set_title('Motion Characteristics: FP vs TP (per Event Type)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('FP/TP Ratio', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_fp_vs_tp.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'scatter_fp_vs_tp.png')}")
    plt.close()


def plot_effect_size_by_event_type(comparison_result, output_dir):
    """
    Bar plot: Effect size (FP/TP ratio) by event type.

    Args:
        comparison_result (dict): Comparison results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    per_event = comparison_result['per_event_comparison']
    effect_data = []

    for event_type, metrics in per_event.items():
        if isinstance(metrics, dict) and 'mean_motion' in metrics:
            comp = metrics['mean_motion']
            effect_data.append({
                'event_type': event_type,
                'ratio': comp['ratio_mean'],
                'p_value': comp['p_value']
            })

    if not effect_data:
        print("⚠ No data for effect size plot")
        return

    df_effect = pd.DataFrame(effect_data).sort_values('ratio', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by significance
    colors = [COLOR_FP if p < 0.05 else COLOR_NEUTRAL for p in df_effect['p_value']]

    bars = ax.bar(range(len(df_effect)), df_effect['ratio'], color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, ratio, p_val in zip(bars, df_effect['ratio'], df_effect['p_value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.2f}x\np={p_val:.2e}',
               ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(df_effect)))
    ax.set_xticklabels(df_effect['event_type'], rotation=45, ha='right')
    ax.set_ylabel('Effect Size (FP/TP Ratio)', fontsize=12, fontweight='bold')
    ax.set_title('Motion Difference by Event Type (FP/TP Ratio)', fontsize=14, fontweight='bold')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1:1 Reference')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_size_bars.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'effect_size_bars.png')}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_distribution(fp_features, tp_features, output_dir):
    """
    Cumulative distribution function (CDF) comparison.

    Args:
        fp_features (pd.DataFrame): FP motion features
        tp_features (pd.DataFrame): TP motion features
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['mean_motion', 'max_motion']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cumulative Distribution Functions (CDF)', fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        fp_data = sorted(fp_features[metric].dropna())
        tp_data = sorted(tp_features[metric].dropna())

        if len(fp_data) == 0 or len(tp_data) == 0:
            continue

        # CDF
        fp_cdf = np.arange(1, len(fp_data)+1) / len(fp_data)
        tp_cdf = np.arange(1, len(tp_data)+1) / len(tp_data)

        ax.plot(fp_data, fp_cdf, linewidth=2.5, label=f'FP (n={len(fp_data)})', color=COLOR_FP)
        ax.plot(tp_data, tp_cdf, linewidth=2.5, label=f'TP (n={len(tp_data)})', color=COLOR_TP)

        ax.set_xlabel('Motion (mG)', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'cumulative_distribution.png')}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Per-Event-Type Detailed Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_event_type_comparison(fp_features, tp_features, event_types, output_dir):
    """
    Create detailed comparison plots for each event type.

    Args:
        fp_features (pd.DataFrame): FP motion features
        tp_features (pd.DataFrame): TP motion features
        event_types (list): Event types to plot
        output_dir (str): Output directory
    """
    for event_type in event_types:
        event_dir = os.path.join(output_dir, event_type.replace(' ', '_'))
        os.makedirs(event_dir, exist_ok=True)

        fp_event = fp_features[fp_features['event_type'] == event_type]
        tp_event = tp_features[tp_features['event_type'] == event_type]

        if len(fp_event) < 3 or len(tp_event) < 3:
            continue

        # Box plot for this event type
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{event_type} - Motion Comparison', fontsize=14, fontweight='bold')

        for ax, metric in zip(axes, ['mean_motion', 'max_motion']):
            fp_data = fp_event[metric].dropna()
            tp_data = tp_event[metric].dropna()

            if len(fp_data) == 0 or len(tp_data) == 0:
                continue

            bp = ax.boxplot([fp_data, tp_data],
                           labels=[f'FP\n(n={len(fp_data)})', f'TP\n(n={len(tp_data)})'],
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

            for patch, color in zip(bp['boxes'], [COLOR_FP, COLOR_TP]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_ylabel('Motion (mG)', fontsize=11)
            ax.set_title(metric, fontsize=12)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(event_dir, f'{event_type}_comparison.png'),
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved event-type plot: {event_type}")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Frequency Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_dominant_frequency_comparison(fp_features, tp_features, output_dir):
    """
    Bar plot comparing dominant frequencies.

    Args:
        fp_features (pd.DataFrame): FP motion features
        tp_features (pd.DataFrame): TP motion features
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'dom_freq' not in fp_features.columns or 'dom_freq' not in tp_features.columns:
        print("⚠ Dominant frequency data not available")
        return

    fp_freq = fp_features['dom_freq'].dropna()
    tp_freq = tp_features['dom_freq'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dominant Frequency Analysis', fontsize=14, fontweight='bold')

    # Histogram
    ax = axes[0]
    ax.hist(fp_freq, bins=30, alpha=0.6, label=f'FP (n={len(fp_freq)})',
            color=COLOR_FP, edgecolor='black', density=True)
    ax.hist(tp_freq, bins=30, alpha=0.6, label=f'TP (n={len(tp_freq)})',
            color=COLOR_TP, edgecolor='black', density=True)
    ax.set_xlabel('Dominant Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of Dominant Frequencies', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([fp_freq, tp_freq],
                    labels=[f'FP\n(n={len(fp_freq)})', f'TP\n(n={len(tp_freq)})'],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], [COLOR_FP, COLOR_TP]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title('Dominant Frequency Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dominant_frequency_comparison.png'),
               dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'dominant_frequency_comparison.png')}")
    plt.close()


if __name__ == "__main__":
    print("Motion Visualization Module")
