"""
motion_comparison_stats.py

Statistical analysis and comparison of motion characteristics
between two ECG datasets.

Provides:
  - Single dataset analysis (mean, median, percentiles, distributions)
  - Cross-dataset comparison (ratios, differences, effect sizes)
  - Per-event-type drill-down analysis
  - Statistical significance testing (parametric + non-parametric)
"""

import numpy as np
import pandas as pd
from scipy import stats
from motion_analysis_utilities import (
    extract_motion_features_batch, compute_distribution_stats, verify_motion_features
)
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Single Dataset Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset_motion(dataset_name, csv_path, data_dir, event_types=None, sample_size=None):
    """
    Comprehensive motion analysis for a single dataset.

    Args:
        dataset_name (str): Name of dataset (e.g., "FP", "TP")
        csv_path (str): Path to summary.csv
        data_dir (str): Root data directory
        event_types (list): Event types to analyze (None = all)
        sample_size (int): Max samples to extract (None = all)

    Returns:
        dict: Analysis results including:
            - features_df: Motion features for each record
            - overall_stats: Dataset-wide statistics
            - per_event_stats: Statistics by event type
            - validation_report: Data quality report
    """
    from motion_analysis_utilities import load_dataset

    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name} Dataset")
    print(f"{'='*80}")

    # Load dataset
    df_records = load_dataset(csv_path, data_dir)

    if event_types:
        df_records = df_records[df_records['event_type'].isin(event_types)]
        print(f"✓ Filtered to {len(df_records)} records of specified event types")

    # Extract motion features
    df_features, failed = extract_motion_features_batch(
        df_records,
        json_col='json_file',
        event_col='event_type',
        sample_size=sample_size,
        stratified=True
    )

    if len(df_features) == 0:
        print("ERROR: No features extracted!")
        return None

    # Overall statistics
    print("\nComputing overall statistics...")
    overall_stats = {}
    for metric in ['mean_motion', 'max_motion', 'std_motion', 'median_motion', 'peak_ratio', 'dom_freq']:
        if metric in df_features.columns:
            data = df_features[metric].dropna()
            overall_stats[metric] = compute_distribution_stats(data)

    # Per-event-type statistics
    print("Computing per-event statistics...")
    per_event_stats = {}
    if 'event_type' in df_features.columns:
        for event_type in sorted(df_features['event_type'].unique()):
            event_data = df_features[df_features['event_type'] == event_type]
            per_event_stats[event_type] = {
                'count': len(event_data),
                'metrics': {}
            }
            for metric in ['mean_motion', 'max_motion', 'std_motion', 'median_motion', 'peak_ratio']:
                if metric in event_data.columns:
                    data = event_data[metric].dropna()
                    if len(data) > 0:
                        per_event_stats[event_type]['metrics'][metric] = compute_distribution_stats(data)

    # Validation
    validation = verify_motion_features(df_features)

    print(f"\n✓ Analysis complete: {len(df_features)} records")

    return {
        'dataset_name': dataset_name,
        'features_df': df_features,
        'overall_stats': overall_stats,
        'per_event_stats': per_event_stats,
        'validation_report': validation,
        'failed_indices': failed
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Dataset Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_datasets_motion(fp_analysis, tp_analysis, event_types=None):
    """
    Comprehensive comparison between two datasets.

    Args:
        fp_analysis (dict): FP dataset analysis results
        tp_analysis (dict): TP dataset analysis results
        event_types (list): Event types to compare (None = common types)

    Returns:
        dict: Comparison results with metrics:
            - overall_comparison: Global FP vs TP statistics
            - per_event_comparison: Per-event-type detailed comparison
            - statistical_tests: Significance tests for each metric
    """
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*80}")

    fp_features = fp_analysis['features_df']
    tp_features = tp_analysis['features_df']

    # Determine event types to compare
    if event_types is None:
        fp_events = set(fp_features['event_type'].unique())
        tp_events = set(tp_features['event_type'].unique())
        event_types = sorted(fp_events & tp_events)

    print(f"\nComparing {len(event_types)} common event types:")
    for et in event_types:
        fp_n = len(fp_features[fp_features['event_type'] == et])
        tp_n = len(tp_features[tp_features['event_type'] == et])
        print(f"  {et:<35} FP: {fp_n:>5} | TP: {tp_n:>5}")

    # Overall comparison (across all records)
    print("\n" + "-"*80)
    print("OVERALL COMPARISON (all records)")
    print("-"*80)

    overall_comparison = {}
    for metric in ['mean_motion', 'max_motion', 'std_motion', 'median_motion', 'peak_ratio', 'dom_freq']:
        if metric not in fp_features.columns or metric not in tp_features.columns:
            continue

        fp_data = fp_features[metric].dropna().values
        tp_data = tp_features[metric].dropna().values

        if len(fp_data) == 0 or len(tp_data) == 0:
            continue

        comparison = _compare_metric(metric, fp_data, tp_data)
        overall_comparison[metric] = comparison

        print(f"\n{metric}:")
        print(f"  FP: mean={comparison['fp_mean']:.4f}, median={comparison['fp_median']:.4f}, "
              f"n={comparison['fp_n']}")
        print(f"  TP: mean={comparison['tp_mean']:.4f}, median={comparison['tp_median']:.4f}, "
              f"n={comparison['tp_n']}")
        print(f"  Ratio (FP/TP):  {comparison['ratio_mean']:.3f}x (means) | "
              f"{comparison['ratio_median']:.3f}x (medians)")
        print(f"  Effect Size (Cohen's d): {comparison['cohens_d']:.3f}")
        print(f"  Statistical Test: p={comparison['p_value']:.4e} "
              f"({'***' if comparison['p_value'] < 0.001 else '**' if comparison['p_value'] < 0.01 else '*' if comparison['p_value'] < 0.05 else 'ns'})")

    # Per-event-type comparison
    print("\n" + "-"*80)
    print("PER-EVENT-TYPE COMPARISON")
    print("-"*80)

    per_event_comparison = {}
    for event_type in event_types:
        fp_event = fp_features[fp_features['event_type'] == event_type]
        tp_event = tp_features[tp_features['event_type'] == event_type]

        if len(fp_event) < 3 or len(tp_event) < 3:
            print(f"\n{event_type}: INSUFFICIENT DATA (FP: {len(fp_event)}, TP: {len(tp_event)})")
            per_event_comparison[event_type] = {'status': 'insufficient_data'}
            continue

        per_event_comparison[event_type] = {}
        print(f"\n{event_type} (FP: {len(fp_event)}, TP: {len(tp_event)}):")

        for metric in ['mean_motion', 'max_motion', 'std_motion', 'median_motion', 'peak_ratio']:
            if metric not in fp_event.columns or metric not in tp_event.columns:
                continue

            fp_data = fp_event[metric].dropna().values
            tp_data = tp_event[metric].dropna().values

            if len(fp_data) < 2 or len(tp_data) < 2:
                continue

            comparison = _compare_metric(metric, fp_data, tp_data)
            per_event_comparison[event_type][metric] = comparison

            print(f"  {metric}: "
                  f"FP={comparison['fp_mean']:.2f} | TP={comparison['tp_mean']:.2f} | "
                  f"Ratio={comparison['ratio_mean']:.2f}x | p={comparison['p_value']:.4e}")

    return {
        'overall_comparison': overall_comparison,
        'per_event_comparison': per_event_comparison,
        'event_types': event_types
    }


def _compare_metric(metric_name, fp_data, tp_data):
    """
    Statistical comparison for a single metric between FP and TP.

    Args:
        metric_name (str): Name of metric (for labeling)
        fp_data (np.ndarray): FP metric values
        tp_data (np.ndarray): TP metric values

    Returns:
        dict: Comprehensive comparison statistics
    """
    # Remove infinite and NaN values
    fp_data = fp_data[np.isfinite(fp_data)]
    tp_data = tp_data[np.isfinite(tp_data)]

    comparison = {
        'metric': metric_name,
        'fp_n': len(fp_data),
        'tp_n': len(tp_data),
        'fp_mean': float(np.mean(fp_data)),
        'fp_median': float(np.median(fp_data)),
        'fp_std': float(np.std(fp_data)),
        'tp_mean': float(np.mean(tp_data)),
        'tp_median': float(np.median(tp_data)),
        'tp_std': float(np.std(tp_data)),
    }

    # Compute ratio
    comparison['ratio_mean'] = comparison['fp_mean'] / (comparison['tp_mean'] + 1e-10)
    comparison['ratio_median'] = comparison['fp_median'] / (comparison['tp_median'] + 1e-10)

    # Absolute difference
    comparison['diff_mean'] = comparison['fp_mean'] - comparison['tp_mean']
    comparison['diff_median'] = comparison['fp_median'] - comparison['tp_median']

    # Percentage difference
    comparison['pct_diff_mean'] = 100 * comparison['diff_mean'] / (comparison['tp_mean'] + 1e-10)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(fp_data)-1) * np.std(fp_data)**2 + (len(tp_data)-1) * np.std(tp_data)**2) /
        (len(fp_data) + len(tp_data) - 2)
    )
    comparison['cohens_d'] = (comparison['fp_mean'] - comparison['tp_mean']) / (pooled_std + 1e-10)

    # Statistical tests
    # Parametric test (t-test, assumes normality)
    t_stat, t_pvalue = stats.ttest_ind(fp_data, tp_data)
    comparison['t_stat'] = float(t_stat)
    comparison['t_pvalue'] = float(t_pvalue)

    # Non-parametric test (Mann-Whitney U, doesn't assume normality)
    u_stat, u_pvalue = stats.mannwhitneyu(fp_data, tp_data, alternative='two-sided')
    comparison['u_stat'] = float(u_stat)
    comparison['u_pvalue'] = float(u_pvalue)

    # Use Mann-Whitney U p-value (more robust for non-normal distributions)
    comparison['p_value'] = float(u_pvalue)

    # Distribution overlap (percentage of TP data within FP IQR)
    fp_q25, fp_q75 = np.percentile(fp_data, [25, 75])
    overlap = np.mean((tp_data >= fp_q25) & (tp_data <= fp_q75))
    comparison['distribution_overlap'] = float(overlap)

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Per-Event-Type Drill-Down
# ─────────────────────────────────────────────────────────────────────────────

def analyze_event_type_motion(analysis_result, event_type, metric='mean_motion'):
    """
    Detailed analysis of motion characteristics for a specific event type.

    Args:
        analysis_result (dict): Dataset analysis result
        event_type (str): Event type to analyze
        metric (str): Motion metric to focus on

    Returns:
        dict: Detailed event-type analysis
    """
    df_features = analysis_result['features_df']
    event_data = df_features[df_features['event_type'] == event_type]

    if len(event_data) == 0:
        return None

    if metric not in event_data.columns:
        return None

    metric_data = event_data[metric].dropna()

    return {
        'event_type': event_type,
        'dataset': analysis_result['dataset_name'],
        'count': len(metric_data),
        'stats': compute_distribution_stats(metric_data),
        'records': event_data
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def create_comparison_table(comparison_result, event_types=None):
    """
    Create a summary table of comparisons.

    Args:
        comparison_result (dict): Comparison results from compare_datasets_motion
        event_types (list): Event types to include (None = all)

    Returns:
        pd.DataFrame: Comparison summary table
    """
    rows = []

    # Overall comparison
    for metric, comp in comparison_result['overall_comparison'].items():
        rows.append({
            'metric': metric,
            'event_type': 'OVERALL',
            'fp_mean': comp['fp_mean'],
            'fp_median': comp['fp_median'],
            'tp_mean': comp['tp_mean'],
            'tp_median': comp['tp_median'],
            'ratio_mean': comp['ratio_mean'],
            'ratio_median': comp['ratio_median'],
            'cohens_d': comp['cohens_d'],
            'p_value': comp['p_value'],
            'fp_n': comp['fp_n'],
            'tp_n': comp['tp_n']
        })

    # Per-event comparison
    event_types_to_process = event_types or comparison_result['per_event_comparison'].keys()

    for event_type in event_types_to_process:
        if event_type not in comparison_result['per_event_comparison']:
            continue

        event_comp = comparison_result['per_event_comparison'][event_type]
        if event_comp.get('status') == 'insufficient_data':
            continue

        for metric, comp in event_comp.items():
            if not isinstance(comp, dict):
                continue

            rows.append({
                'metric': metric,
                'event_type': event_type,
                'fp_mean': comp['fp_mean'],
                'fp_median': comp['fp_median'],
                'tp_mean': comp['tp_mean'],
                'tp_median': comp['tp_median'],
                'ratio_mean': comp['ratio_mean'],
                'ratio_median': comp['ratio_median'],
                'cohens_d': comp['cohens_d'],
                'p_value': comp['p_value'],
                'fp_n': comp['fp_n'],
                'tp_n': comp['tp_n']
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Motion Comparison Statistics Module")
