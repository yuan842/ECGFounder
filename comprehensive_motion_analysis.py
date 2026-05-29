#!/usr/bin/env python3
"""
comprehensive_motion_analysis.py

Main orchestration script for comprehensive motion/accelerometer analysis
comparing ECG-FP-Doctor-Removed1 and ECG-TP_REX datasets.

Workflow:
  1. Load both datasets
  2. Extract motion features
  3. Compute statistics and comparisons
  4. Generate visualizations
  5. Create comprehensive report
  6. Save all outputs to organized directory structure
"""

import os
import sys
import argparse
from pathlib import Path

from motion_analysis_utilities import load_dataset
from motion_comparison_stats import (
    analyze_dataset_motion, compare_datasets_motion, create_comparison_table
)
from motion_visualization import (
    plot_overall_distribution_comparison,
    plot_box_and_violin_comparison,
    plot_scatter_fp_vs_tp,
    plot_effect_size_by_event_type,
    plot_cumulative_distribution,
    plot_per_event_type_comparison,
    plot_dominant_frequency_comparison
)
from motion_analysis_report import generate_motion_analysis_report

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dataset paths (update these to match your environment)
FP_CSV = "./data/ecg_fp_doctor removed1/summary.csv"
FP_DIR = "./data/ecg_fp_doctor removed1"

TP_CSV = "./data/ecg-tp_rex/summary.csv"
TP_DIR = "./data/ecg-tp_rex"

OUTPUT_DIR = "./res/motion_analysis"

# Analysis parameters
FP_SAMPLE_SIZE = 500  # Sample up to 500 FP records per event type (stratified)
TP_SAMPLE_SIZE = None  # Use all TP records

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run comprehensive motion analysis."""
    print("="*80)
    print("COMPREHENSIVE MOTION/ACCELEROMETER ANALYSIS")
    print("ECG-FP-Doctor-Removed1 vs ECG-TP_REX Comparison")
    print("="*80)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_DIR, 'figures')
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # ── Step 1: Analyze FP Dataset ────────────────────────────────────────────
    print("STEP 1: Analyzing False Positive (FP) Dataset")
    print("-" * 80)

    if not os.path.exists(FP_CSV):
        print(f"ERROR: FP dataset CSV not found: {FP_CSV}")
        return 1

    fp_analysis = analyze_dataset_motion(
        dataset_name="FP (False Positives)",
        csv_path=FP_CSV,
        data_dir=FP_DIR,
        event_types=None,
        sample_size=FP_SAMPLE_SIZE
    )

    if fp_analysis is None:
        print("ERROR: Failed to analyze FP dataset")
        return 1

    # Save FP features
    fp_csv_path = os.path.join(data_dir, 'fp_motion_features.csv')
    fp_analysis['features_df'].to_csv(fp_csv_path, index=False)
    print(f"✓ Saved FP features to {fp_csv_path}\n")

    # ── Step 2: Analyze TP Dataset ────────────────────────────────────────────
    print("STEP 2: Analyzing True Positive (TP) Dataset")
    print("-" * 80)

    if not os.path.exists(TP_CSV):
        print(f"ERROR: TP dataset CSV not found: {TP_CSV}")
        return 1

    tp_analysis = analyze_dataset_motion(
        dataset_name="TP (True Positives)",
        csv_path=TP_CSV,
        data_dir=TP_DIR,
        event_types=None,
        sample_size=TP_SAMPLE_SIZE
    )

    if tp_analysis is None:
        print("ERROR: Failed to analyze TP dataset")
        return 1

    # Save TP features
    tp_csv_path = os.path.join(data_dir, 'tp_motion_features.csv')
    tp_analysis['features_df'].to_csv(tp_csv_path, index=False)
    print(f"✓ Saved TP features to {tp_csv_path}\n")

    # ── Step 3: Cross-Dataset Comparison ──────────────────────────────────────
    print("STEP 3: Cross-Dataset Comparison")
    print("-" * 80)

    comparison_result = compare_datasets_motion(fp_analysis, tp_analysis)

    if comparison_result is None:
        print("ERROR: Failed to compare datasets")
        return 1

    # Save comparison results
    comparison_table = create_comparison_table(comparison_result)
    comparison_csv = os.path.join(data_dir, 'comparison_stats.csv')
    comparison_table.to_csv(comparison_csv, index=False)
    print(f"✓ Saved comparison stats to {comparison_csv}\n")

    # ── Step 4: Visualizations ────────────────────────────────────────────────
    print("STEP 4: Generating Visualizations")
    print("-" * 80)

    fp_features = fp_analysis['features_df']
    tp_features = tp_analysis['features_df']

    # Overall distribution comparisons
    plot_overall_distribution_comparison(fp_features, tp_features, figures_dir)
    plot_box_and_violin_comparison(fp_features, tp_features, figures_dir)
    plot_cumulative_distribution(fp_features, tp_features, figures_dir)

    # Cross-dataset analysis
    plot_scatter_fp_vs_tp(comparison_result, figures_dir)
    plot_effect_size_by_event_type(comparison_result, figures_dir)

    # Per-event-type analysis
    event_types = comparison_result.get('event_types', [])
    if event_types:
        plot_per_event_type_comparison(fp_features, tp_features, event_types,
                                       os.path.join(figures_dir, 'per_event_type'))

    # Frequency analysis
    if 'dom_freq' in fp_features.columns:
        plot_dominant_frequency_comparison(fp_features, tp_features,
                                          os.path.join(figures_dir, 'frequency_analysis'))

    print(f"✓ Visualizations saved to {figures_dir}\n")

    # ── Step 5: Generate Report ───────────────────────────────────────────────
    print("STEP 5: Generating Report")
    print("-" * 80)

    report_path = os.path.join(OUTPUT_DIR, 'MOTION_ANALYSIS_REPORT.md')
    generate_motion_analysis_report(
        fp_analysis, tp_analysis, comparison_result,
        report_path, comparison_table
    )

    print(f"✓ Report saved to {report_path}\n")

    # ── Step 6: Summary Statistics ────────────────────────────────────────────
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nOutput Structure:")
    print(f"  Data files:  {data_dir}/")
    print(f"  Figures:     {figures_dir}/")
    print(f"  Report:      {report_path}")

    print(f"\nGenerated Files:")
    print(f"  • fp_motion_features.csv - FP motion metrics for all records")
    print(f"  • tp_motion_features.csv - TP motion metrics for all records")
    print(f"  • comparison_stats.csv - Cross-dataset comparison summary")
    print(f"  • Visualizations (PNG/PDF formats)")
    print(f"  • MOTION_ANALYSIS_REPORT.md - Comprehensive markdown report")

    print(f"\nKey Findings:")
    if 'mean_motion' in fp_features.columns and 'mean_motion' in tp_features.columns:
        fp_mean = fp_features['mean_motion'].mean()
        tp_mean = tp_features['mean_motion'].mean()
        ratio = fp_mean / (tp_mean + 1e-10)
        print(f"  • FP mean motion: {fp_mean:.2f} mG")
        print(f"  • TP mean motion: {tp_mean:.2f} mG")
        print(f"  • Ratio (FP/TP): {ratio:.2f}x")

        # Count significant metrics
        overall_comp = comparison_result.get('overall_comparison', {})
        sig_count = sum(1 for comp in overall_comp.values() if comp.get('p_value', 1) < 0.05)
        print(f"  • Statistically significant metrics: {sig_count}/{len(overall_comp)}")

    print(f"\n✓ Analysis successfully completed!\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive motion/accelerometer analysis for ECG datasets'
    )

    parser.add_argument('--fp-csv', default=FP_CSV,
                       help='Path to FP dataset summary.csv')
    parser.add_argument('--fp-dir', default=FP_DIR,
                       help='Path to FP data directory')
    parser.add_argument('--tp-csv', default=TP_CSV,
                       help='Path to TP dataset summary.csv')
    parser.add_argument('--tp-dir', default=TP_DIR,
                       help='Path to TP data directory')
    parser.add_argument('--output', default=OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--fp-sample', type=int, default=FP_SAMPLE_SIZE,
                       help='Max FP samples to analyze (None = all)')
    parser.add_argument('--tp-sample', type=int, default=TP_SAMPLE_SIZE,
                       help='Max TP samples to analyze (None = all)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Update configuration from CLI args
    FP_CSV = args.fp_csv
    FP_DIR = args.fp_dir
    TP_CSV = args.tp_csv
    TP_DIR = args.tp_dir
    OUTPUT_DIR = args.output
    FP_SAMPLE_SIZE = args.fp_sample
    TP_SAMPLE_SIZE = args.tp_sample

    exit_code = main()
    sys.exit(exit_code)
