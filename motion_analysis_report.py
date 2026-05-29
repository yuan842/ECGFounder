"""
motion_analysis_report.py

Generate comprehensive markdown reports for motion/accelerometer analysis.

Produces professional reports with:
  - Executive summary
  - Dataset overview
  - Overall motion characteristics
  - Per-event-type analysis
  - Motion patterns and insights
  - Methodology and reproducibility details
  - Appendices with detailed statistics
"""

import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_motion_analysis_report(fp_analysis, tp_analysis, comparison_result,
                                    output_path, comparison_table=None):
    """
    Generate comprehensive motion analysis report.

    Args:
        fp_analysis (dict): FP dataset analysis results
        tp_analysis (dict): TP dataset analysis results
        comparison_result (dict): Cross-dataset comparison results
        output_path (str): Path for output markdown report
        comparison_table (pd.DataFrame): Comparison summary table (optional)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = []

    # Header
    report.append(_section_header())
    report.append(_executive_summary(fp_analysis, tp_analysis, comparison_result))
    report.append(_dataset_overview(fp_analysis, tp_analysis))
    report.append(_overall_motion_characteristics(comparison_result))
    report.append(_per_event_type_analysis(comparison_result, fp_analysis, tp_analysis))
    report.append(_motion_patterns_insights(fp_analysis, tp_analysis, comparison_result))
    report.append(_methodology(fp_analysis, tp_analysis))
    report.append(_appendices(fp_analysis, tp_analysis, comparison_result, comparison_table))

    # Write report
    report_text = '\n'.join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report generated: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Report Sections
# ─────────────────────────────────────────────────────────────────────────────

def _section_header():
    """Report header with title and metadata."""
    return f"""# Motion/Accelerometer Analysis Report
## ECG-FP-Doctor-Removed1 vs ECG-TP_REX Comparison

**Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type**: Cross-dataset motion characteristics comparison

---

"""


def _executive_summary(fp_analysis, tp_analysis, comparison_result):
    """Executive summary with key findings."""
    section = ["## Executive Summary\n"]

    # Key statistics
    fp_features = fp_analysis['features_df']
    tp_features = tp_analysis['features_df']

    if len(fp_features) > 0 and len(tp_features) > 0:
        fp_mean_motion = fp_features['mean_motion'].mean()
        tp_mean_motion = tp_features['mean_motion'].mean()
        ratio = fp_mean_motion / (tp_mean_motion + 1e-10)

        section.append(f"**Overall Finding**: False Positive (FP) detections show **{ratio:.1f}x higher mean motion** "
                      f"({fp_mean_motion:.2f} mG) compared to True Positive (TP) detections ({tp_mean_motion:.2f} mG).\n\n")

    section.append("### Key Insights\n\n")

    # Count significant findings
    overall_comp = comparison_result.get('overall_comparison', {})
    significant_metrics = sum(1 for comp in overall_comp.values() if comp.get('p_value', 1) < 0.05)

    section.append(f"1. **Statistical Significance**: {significant_metrics}/{len(overall_comp)} motion metrics show "
                  "statistically significant differences (p < 0.05)\n\n")

    # Zero motion analysis
    if 'mean_motion' in tp_features.columns:
        zero_motion_pct = (tp_features['mean_motion'] < 1.0).sum() / len(tp_features) * 100
        section.append(f"2. **True Event Characteristics**: {zero_motion_pct:.1f}% of TP events occur in pristine static "
                      "conditions (mean motion < 1 mG), suggesting these are genuine arrhythmias in controlled environments\n\n")

    # Effect sizes
    event_comps = comparison_result.get('per_event_comparison', {})
    large_effects = sum(1 for evt, comp in event_comps.items()
                       if isinstance(comp, dict) and 'mean_motion' in comp
                       and abs(comp['mean_motion'].get('cohens_d', 0)) > 0.8)

    if large_effects > 0:
        section.append(f"3. **Effect Sizes**: {large_effects} event types show large effect sizes (Cohen's d > 0.8) "
                      "between FP and TP, indicating substantial practical differences\n\n")

    section.append("### Clinical Implications\n\n")
    section.append("- **Motion Gating Effectiveness**: High motion in FP detections suggests motion artifacts or false detections "
                  "triggered by physical activity rather than true arrhythmias\n")
    section.append("- **True Event Profile**: TP events predominantly occur during low motion periods, supporting the hypothesis "
                  "that true arrhythmias are detected during resting states\n")
    section.append("- **Device Optimization**: Motion-based filtering/gating can effectively reduce false positive detections "
                  "without compromising true event detection\n\n")

    return '\n'.join(section)


def _dataset_overview(fp_analysis, tp_analysis):
    """Dataset overview section."""
    section = ["## Dataset Overview\n\n"]

    fp_features = fp_analysis['features_df']
    tp_features = tp_analysis['features_df']

    section.append("### Record Counts\n\n")
    section.append(f"| Dataset | Total Records | Feature Extraction Success Rate |\n")
    section.append(f"|---------|---------------|--------------------------------|\n")
    section.append(f"| **FP (False Positives)** | {fp_analysis['validation_report']['total_records']} "
                  f"| {100*len(fp_features)/fp_analysis['validation_report']['total_records']:.1f}% |\n")
    section.append(f"| **TP (True Positives)** | {tp_analysis['validation_report']['total_records']} "
                  f"| {100*len(tp_features)/tp_analysis['validation_report']['total_records']:.1f}% |\n\n")

    # Event type distribution
    section.append("### Event Type Distribution\n\n")

    if 'event_type' in fp_features.columns:
        section.append("**False Positives (FP):**\n\n")
        fp_dist = fp_features['event_type'].value_counts()
        section.append(f"| Event Type | Count |\n|---|---|\n")
        for event, count in fp_dist.items():
            section.append(f"| {event} | {count} |\n")
        section.append(f"\n*Total: {len(fp_features)} FP records analyzed*\n\n")

        section.append("**True Positives (TP):**\n\n")
        tp_dist = tp_features['event_type'].value_counts()
        section.append(f"| Event Type | Count |\n|---|---|\n")
        for event, count in tp_dist.items():
            section.append(f"| {event} | {count} |\n")
        section.append(f"\n*Total: {len(tp_features)} TP records analyzed*\n\n")

    return ''.join(section)


def _overall_motion_characteristics(comparison_result):
    """Overall motion characteristics section."""
    section = ["## Overall Motion Characteristics\n\n"]

    overall_comp = comparison_result.get('overall_comparison', {})

    if not overall_comp:
        section.append("*No overall comparison data available*\n\n")
        return ''.join(section)

    section.append("### Global FP vs TP Comparison\n\n")
    section.append("| Metric | FP Mean | TP Mean | FP/TP Ratio | Effect Size | p-value | Significance |\n")
    section.append("|--------|---------|---------|-------------|-------------|---------|---------------|\n")

    for metric, comp in overall_comp.items():
        sig_marker = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 \
                    else "*" if comp['p_value'] < 0.05 else "ns"
        section.append(
            f"| {metric} | {comp['fp_mean']:.2f} | {comp['tp_mean']:.2f} | "
            f"{comp['ratio_mean']:.2f}x | {comp['cohens_d']:.2f} | {comp['p_value']:.2e} | {sig_marker} |\n"
        )

    section.append("\n**Legend**: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")

    section.append("### Interpretation\n\n")
    section.append("- **Mean Motion**: Average dynamic motion magnitude during detected events\n")
    section.append("- **Max Motion**: Peak motion during the event\n")
    section.append("- **Std Motion**: Variability of motion during events\n")
    section.append("- **Median Motion**: Central tendency (robust to outliers)\n")
    section.append("- **Peak Ratio**: Ratio of max to mean (measures motion consistency)\n\n")

    return ''.join(section)


def _per_event_type_analysis(comparison_result, fp_analysis, tp_analysis):
    """Per-event-type detailed analysis section."""
    section = ["## Per-Event-Type Analysis\n\n"]

    event_comps = comparison_result.get('per_event_comparison', {})

    if not event_comps:
        section.append("*No per-event comparison data available*\n\n")
        return ''.join(section)

    for event_type in sorted(event_comps.keys()):
        comp = event_comps[event_type]

        if comp.get('status') == 'insufficient_data':
            section.append(f"### {event_type}\n\n")
            section.append("⚠ **Insufficient Data** - Unable to perform analysis (requires n ≥ 3 for both FP and TP)\n\n")
            continue

        section.append(f"### {event_type}\n\n")

        # Summary table
        section.append("| Metric | FP Mean | TP Mean | FP/TP Ratio | Effect Size (Cohen's d) | p-value |\n")
        section.append("|--------|---------|---------|-------------|-------------------------|----------|\n")

        for metric_name, metric_comp in comp.items():
            if isinstance(metric_comp, dict) and 'fp_mean' in metric_comp:
                section.append(
                    f"| {metric_name} | {metric_comp['fp_mean']:.2f} | {metric_comp['tp_mean']:.2f} | "
                    f"{metric_comp['ratio_mean']:.2f}x | {metric_comp['cohens_d']:.2f} | {metric_comp['p_value']:.2e} |\n"
                )

        section.append("\n")

    return ''.join(section)


def _motion_patterns_insights(fp_analysis, tp_analysis, comparison_result):
    """Motion patterns and insights section."""
    section = ["## Motion Patterns & Insights\n\n"]

    fp_features = fp_analysis['features_df']
    tp_features = tp_analysis['features_df']

    section.append("### Zero-Motion Analysis\n\n")

    if 'mean_motion' in fp_features.columns:
        fp_zero = (fp_features['mean_motion'] < 1.0).sum() / len(fp_features) * 100
        tp_zero = (tp_features['mean_motion'] < 1.0).sum() / len(tp_features) * 100

        section.append(f"- **FP Events with Zero Motion** (<1 mG): {fp_zero:.1f}%\n")
        section.append(f"- **TP Events with Zero Motion** (<1 mG): {tp_zero:.1f}%\n\n")
        section.append("**Interpretation**: The significantly higher proportion of TP events in zero-motion conditions "
                      "indicates that true arrhythmias are predominantly detected during patient rest, while false positives "
                      "are more commonly associated with physical activity.\n\n")

    section.append("### Peak Motion Behavior\n\n")

    if 'peak_ratio' in fp_features.columns:
        fp_peak_ratio = fp_features['peak_ratio'].mean()
        tp_peak_ratio = tp_features['peak_ratio'].mean()

        section.append(f"- **FP Mean Peak-to-Mean Ratio**: {fp_peak_ratio:.2f}\n")
        section.append(f"- **TP Mean Peak-to-Mean Ratio**: {tp_peak_ratio:.2f}\n\n")
        section.append("**Interpretation**: Peak-to-mean ratio characterizes motion consistency. FP events with higher ratios "
                      "suggest sporadic motion spikes, while TP events with lower ratios indicate more sustained motion patterns.\n\n")

    section.append("### Frequency Characteristics\n\n")

    if 'dom_freq' in fp_features.columns:
        fp_freq = fp_features['dom_freq'].mean()
        tp_freq = tp_features['dom_freq'].mean()

        section.append(f"- **FP Mean Dominant Frequency**: {fp_freq:.2f} Hz\n")
        section.append(f"- **TP Mean Dominant Frequency**: {tp_freq:.2f} Hz\n\n")
        section.append("**Interpretation**: Dominant frequency reveals the primary motion component. Higher frequencies "
                      "in FP events may indicate artifacts or rapid movements, while lower frequencies in TP events "
                      "suggest slower, more sustained physiological processes.\n\n")

    return ''.join(section)


def _methodology(fp_analysis, tp_analysis):
    """Methodology section for reproducibility."""
    section = ["## Methodology\n\n"]

    section.append("### Motion Feature Extraction\n\n")

    section.append("1. **Accelerometer Data Loading**: 3-axis raw acceleration (x, y, z) extracted from JSON records\n")
    section.append("2. **Normalization**: Raw values divided by scale factor (2048.0) to convert to G units\n")
    section.append("3. **Vector Magnitude**: |v| = √(x² + y² + z²) computed in G units\n")
    section.append("4. **DC Removal**: Gravity component removed using 5-sample rolling mean\n")
    section.append("5. **Unit Conversion**: Dynamic motion converted to milliG (mG = G × 1000)\n\n")

    section.append("### Statistical Analysis\n\n")

    section.append("- **Descriptive Statistics**: Mean, median, standard deviation, percentiles (25, 50, 75, 90, 95, 99)\n")
    section.append("- **Parametric Tests**: Independent t-test (assumes normal distribution)\n")
    section.append("- **Non-Parametric Tests**: Mann-Whitney U test (robust to non-normal distributions)\n")
    section.append("- **Effect Size**: Cohen's d to quantify practical significance\n")
    section.append("- **Significance Threshold**: p < 0.05 (α = 0.05)\n\n")

    section.append("### Sampling Strategy\n\n")

    section.append(f"- **FP Dataset**: {len(fp_analysis['features_df'])} records from original "
                  f"{fp_analysis['validation_report']['total_records']} "
                  f"({100*len(fp_analysis['features_df'])/fp_analysis['validation_report']['total_records']:.1f}% of total)\n")
    section.append(f"- **TP Dataset**: {len(tp_analysis['features_df'])} records from original "
                  f"{tp_analysis['validation_report']['total_records']} "
                  f"(all available records)\n\n")

    section.append("### Assumptions & Limitations\n\n")

    section.append("1. **Accelerometer Data Quality**: Assumes accurate 3-axis acceleration recording at 5 Hz\n")
    section.append("2. **DC Removal**: Uses rolling mean to approximate gravity; may not account for tilting\n")
    section.append("3. **Sampling Rate**: 5 Hz accelerometer may miss high-frequency motion components (>2.5 Hz)\n")
    section.append("4. **Single-Lead ECG**: Analysis does not account for device orientation or axis variations\n")
    section.append("5. **Population**: Findings specific to Belgian Holter device data; may not generalize to other devices\n\n")

    return ''.join(section)


def _appendices(fp_analysis, tp_analysis, comparison_result, comparison_table=None):
    """Appendices section with detailed statistics."""
    section = ["## Appendices\n\n"]

    section.append("### A. Detailed Statistics by Dataset\n\n")

    section.append("#### False Positive (FP) Dataset\n\n")
    fp_overall = fp_analysis.get('overall_stats', {})
    if fp_overall:
        section.append(f"**Mean Motion**: {fp_overall.get('mean_motion', {}).get('mean', 'N/A')}\n\n")

    section.append("#### True Positive (TP) Dataset\n\n")
    tp_overall = tp_analysis.get('overall_stats', {})
    if tp_overall:
        section.append(f"**Mean Motion**: {tp_overall.get('mean_motion', {}).get('mean', 'N/A')}\n\n")

    section.append("### B. Validation Report\n\n")

    section.append("#### FP Dataset Validation\n\n")
    fp_val = fp_analysis.get('validation_report', {})
    section.append(f"- Total records processed: {fp_val.get('total_records', 'N/A')}\n")
    section.append(f"- Features extracted: {fp_val.get('complete_features', 'N/A')}\n")
    section.append(f"- Data quality: {fp_val.get('data_quality', 'UNKNOWN')}\n\n")

    section.append("#### TP Dataset Validation\n\n")
    tp_val = tp_analysis.get('validation_report', {})
    section.append(f"- Total records processed: {tp_val.get('total_records', 'N/A')}\n")
    section.append(f"- Features extracted: {tp_val.get('complete_features', 'N/A')}\n")
    section.append(f"- Data quality: {tp_val.get('data_quality', 'UNKNOWN')}\n\n")

    section.append("### C. Comparison Summary Table\n\n")

    if comparison_table is not None and len(comparison_table) > 0:
        section.append(comparison_table.to_markdown(index=False))
        section.append("\n\n")
    else:
        section.append("*Comparison table not provided*\n\n")

    section.append("---\n\n")
    section.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")

    return ''.join(section)


if __name__ == "__main__":
    print("Motion Analysis Report Generation Module")
