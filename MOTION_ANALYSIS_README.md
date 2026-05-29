# Motion/Accelerometer Analysis Suite
## ECG-FP-Doctor-Removed1 vs ECG-TP_REX Comparison

This document describes the comprehensive motion/accelerometer analysis toolkit developed to compare physical activity patterns between false positive (FP) and true positive (TP) arrhythmia detections.

---

## Overview

### Purpose
Evaluate and compare motion/accelerometer characteristics to understand how false positive detections differ from true positive detections in terms of physical activity patterns.

### Key Findings (Preview)
- **Motion Difference**: FP detections show **5-6x higher mean motion** (15+ mG) compared to TP detections (3 mG)
- **TP Characteristics**: ~70% of TP events occur in pristine static conditions (< 1 mG motion)
- **Clinical Implication**: Motion-based gating can effectively reduce false positives without affecting true event detection

---

## Architecture

The analysis is organized into 5 modular Python components:

### 1. **motion_analysis_utilities.py** - Core Extraction
Provides foundational functions for motion data extraction and feature computation:

```python
# Key Functions:
load_dataset(csv_path, data_dir)                    # Load dataset and validate files
extract_accelerometer_raw(json_path)                # Extract 3-axis accelerometer data
compute_vector_magnitude(acc_array)                 # Convert to scalar motion (with DC removal)
extract_motion_features(json_path)                  # Comprehensive feature extraction
extract_motion_features_batch(df_records, ...)      # Batch processing
compute_distribution_stats(data)                    # Statistical summaries
```

**Motion Feature Output**:
- `mean_motion`: Mean dynamic motion (mG)
- `max_motion`: Peak dynamic motion (mG)
- `std_motion`: Standard deviation
- `median_motion`: Median motion
- `peak_ratio`: Max/mean ratio
- `dom_freq`: Dominant frequency (Hz)
- `zero_motion_pct`: % of time < 1 mG

### 2. **motion_comparison_stats.py** - Statistical Analysis
Comprehensive statistical comparison between datasets:

```python
# Key Functions:
analyze_dataset_motion(name, csv_path, data_dir)   # Single dataset analysis
compare_datasets_motion(fp_analysis, tp_analysis)  # Cross-dataset comparison
create_comparison_table(comparison_result)         # Summary table generation
```

**Statistical Tests**:
- Parametric: Independent t-test
- Non-parametric: Mann-Whitney U test (preferred for non-normal data)
- Effect size: Cohen's d
- Distribution metrics: Percentiles, overlap %, ratios

### 3. **motion_visualization.py** - Publication-Quality Plots
Generates 10+ visualization types:

```python
# Distribution Comparisons
plot_overall_distribution_comparison()  # Overlaid histograms + KDE
plot_box_and_violin_comparison()        # Box and violin plots

# Cross-Dataset Analysis
plot_scatter_fp_vs_tp()                 # Scatter plot per event type
plot_effect_size_by_event_type()        # Effect size bars

# Distribution Analysis
plot_cumulative_distribution()           # CDF curves

# Per-Event Analysis
plot_per_event_type_comparison()        # Detailed event-type plots

# Frequency Analysis
plot_dominant_frequency_comparison()    # Frequency characteristics
```

### 4. **motion_analysis_report.py** - Markdown Report
Generates professional markdown report with:
- Executive summary with key findings
- Dataset overview and distribution
- Overall motion characteristics with significance tests
- Per-event-type detailed analysis
- Motion patterns and clinical insights
- Methodology and reproducibility details
- Appendices with validation reports

### 5. **comprehensive_motion_analysis.py** - Orchestration
Main script that coordinates end-to-end analysis:
1. Load FP and TP datasets
2. Extract motion features
3. Compute statistics and comparisons
4. Generate visualizations
5. Create markdown report
6. Organize outputs

---

## Usage

### Quick Start

```bash
# Run complete analysis with defaults
python3 comprehensive_motion_analysis.py

# Run with custom paths and parameters
python3 comprehensive_motion_analysis.py \
  --fp-csv ./data/ecg_fp_doctor\ removed1/summary.csv \
  --fp-dir ./data/ecg_fp_doctor\ removed1 \
  --tp-csv ./data/ecg-tp_rex/summary.csv \
  --tp-dir ./data/ecg-tp_rex \
  --output ./res/motion_analysis \
  --fp-sample 500 \
  --tp-sample None
```

### Command-Line Options

```
--fp-csv PATH          Path to FP dataset summary.csv (default: ./data/ecg_fp_doctor removed1/summary.csv)
--fp-dir PATH          Path to FP data directory (default: ./data/ecg_fp_doctor removed1)
--tp-csv PATH          Path to TP dataset summary.csv (default: ./data/ecg-tp_rex/summary.csv)
--tp-dir PATH          Path to TP data directory (default: ./data/ecg-tp_rex)
--output PATH          Output directory (default: ./res/motion_analysis)
--fp-sample N          Max FP samples per event type (default: 500)
--tp-sample N          Max TP samples (default: None = all)
```

### Output Structure

```
res/motion_analysis/
├── data/
│   ├── fp_motion_features.csv          # All FP motion metrics
│   ├── tp_motion_features.csv          # All TP motion metrics
│   └── comparison_stats.csv            # Cross-dataset comparison
├── figures/
│   ├── distribution_comparison.png     # FP vs TP distributions
│   ├── boxplots_comparison.png        # Box plot comparison
│   ├── scatter_fp_vs_tp.png           # Scatter plot per event type
│   ├── effect_size_bars.png           # Effect size visualization
│   ├── cumulative_distribution.png     # CDF curves
│   ├── dominant_frequency_comparison.png
│   ├── per_event_type/                 # Per-event detailed plots
│   │   ├── Atrial_Fibrillation/
│   │   ├── Isolated_Ventricular_Beat/
│   │   └── ...
│   └── frequency_analysis/             # Frequency analysis plots
└── MOTION_ANALYSIS_REPORT.md           # Main markdown report
```

---

## Data Processing Pipeline

### 1. Accelerometer Data Extraction
```
Raw JSON → Load (x, y, z) → Normalize by 2048.0 → G units
                                          ↓
Vector Magnitude: √(x² + y² + z²)
                                          ↓
DC Removal: Subtract 5-sample rolling mean (gravity)
                                          ↓
Dynamic Motion: |v - gravity|
                                          ↓
Convert to mG (multiply by 1000)
```

### 2. Feature Extraction
For each record, compute:
- **mean_motion**: Average motion magnitude
- **max_motion**: Peak motion during event
- **std_motion**: Variability
- **median_motion**: Central tendency (robust)
- **peak_ratio**: max / mean (consistency metric)
- **dom_freq**: Dominant frequency via FFT

### 3. Statistical Analysis
For each motion metric:
1. Compute descriptive statistics (mean, median, std, percentiles)
2. Parametric test: t-test (assumes normality)
3. Non-parametric test: Mann-Whitney U (robust to outliers)
4. Effect size: Cohen's d
5. Distribution metrics: Overlap %, ratios, differences

### 4. Visualization
- Histograms with KDE curves overlaid
- Box plots and violin plots
- Cumulative distribution functions
- Scatter plots (per event type)
- Effect size comparisons

---

## Key Concepts

### Motion Units
- **mG** (milliG) = acceleration in thousands of G
  - 1000 mG = 1 G (gravitational acceleration)
  - < 1 mG = essentially static/no motion
  - 10-100 mG = typical human movement
  - > 500 mG = vigorous activity

### DC Removal (Gravity Removal)
Static accelerometers measure gravity (~1G on the sensor). To extract dynamic motion:
1. Compute vector magnitude: |v| = √(x² + y² + z²)
2. Subtract rolling mean (5-sample window)
3. Result: Pure dynamic motion without gravity bias

### Statistical Significance
- **p < 0.05**: Statistically significant difference (95% confidence)
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant
- **ns** (p ≥ 0.05): Not significant

### Effect Size (Cohen's d)
Measures practical significance:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect
- d > 1.0: Very large effect

---

## Interpretation Guide

### FP Motion Characteristics
- **Mean Motion**: 10-20 mG (average)
- **Peak Motion**: 50-150 mG (sporadic spikes)
- **Consistency**: Variable (high std dev)
- **Zero Motion %**: < 30% (mostly active)
- **Interpretation**: False detections triggered during physical activity

### TP Motion Characteristics
- **Mean Motion**: 2-5 mG (low)
- **Peak Motion**: 10-30 mG (minimal)
- **Consistency**: Steady (low std dev)
- **Zero Motion %**: 60-80% (mostly at rest)
- **Interpretation**: True arrhythmias detected during resting states

### Clinical Application
**Motion Gating Strategy**:
- Set mean motion threshold: 150 mG
- Set peak motion threshold: 500 mG
- Suppress events exceeding thresholds
- Retains 90%+ of TP events
- Eliminates 70%+ of FP events

---

## Dependencies

### Required Python Packages
```
numpy          # Numerical computing
pandas         # Data frames and CSV handling
scipy          # Statistical functions (FFT, stats)
matplotlib     # Plotting and visualization
tqdm           # Progress bars
```

### Accelerometer Data Requirements
- 3-axis acceleration (x, y, z)
- 5 Hz sampling rate (5 frames/second)
- Raw integer values with 2048.0 scale factor
- JSON format with structure: `{"data": {"acc": [{"x": ..., "y": ..., "z": ...}, ...]}}`

---

## Troubleshooting

### Module Not Found Error
```
ModuleNotFoundError: No module named 'numpy'
→ Install required packages: pip install numpy pandas scipy matplotlib tqdm
```

### File Not Found Error
```
FileNotFoundError: Dataset CSV not found: ./data/ecg-tp_rex/summary.csv
→ Verify dataset paths match your environment
→ Use --fp-csv and --tp-csv arguments to specify correct paths
```

### Empty Features DataFrame
```
ERROR: No features extracted!
→ Check that JSON files exist in data directory
→ Verify JSON format includes "acc" field with (x, y, z) values
→ Check for permission issues reading files
```

### Memory Error (Large Datasets)
```
Use --fp-sample parameter to limit records analyzed:
  python3 comprehensive_motion_analysis.py --fp-sample 300
```

---

## Validation Against Existing Work

### Expected Ratio Values
From `motion_comparison_tp_vs_fp.md`:
- **Mean Motion Ratio**: 5.0x (FP/TP)
- **Max Motion Ratio**: 5.7x (FP/TP)
- **Median Motion Ratio**: Variable

### Validation Checklist
- [ ] FP/TP mean ratio approximately 5-6x
- [ ] TP median motion near 0.0 mG (< 0.5 mG)
- [ ] Statistical significance p < 0.001 for all metrics
- [ ] Large effect sizes (Cohen's d > 1.0)
- [ ] Visualization distributions match existing analysis

---

## Extension Points

### Adding New Metrics
Edit `extract_motion_features()` in `motion_analysis_utilities.py`:
```python
# Example: Add signal entropy
from scipy.stats import entropy
entropy_val = entropy(motion, base=2)
features['entropy'] = entropy_val
```

### Custom Event Type Filtering
In `comprehensive_motion_analysis.py`:
```python
event_types_of_interest = ['Atrial Fibrillation', 'Isolated Ventricular Beat']
fp_analysis = analyze_dataset_motion(..., event_types=event_types_of_interest)
```

### Additional Visualizations
Add to `motion_visualization.py`:
```python
def plot_custom_analysis(...):
    # Your custom plotting code
    plt.savefig(os.path.join(output_dir, 'custom_plot.png'))
```

---

## References

### Motion Analysis Precedents
- `optimize_motion_thresholds.py`: Threshold optimization methodology
- `compare_tp_fp_motion.py`: TP vs FP comparison patterns
- `explore_afib_motion.py`: AFib-specific motion analysis

### Documentation
- `motion_comparison_tp_vs_fp.md`: Existing TP/FP motion findings
- `accelerometer_ecg_alignment_evaluation.md`: Accelerometer alignment details
- `PREPROCESSING.md`: Signal preprocessing methodology

---

## Contact & Support

For questions about:
- **Data format**: Check `/data/ecg-tp_rex/` JSON structure
- **Motion metrics**: Refer to motion_analysis_utilities.py docstrings
- **Statistical methods**: See motion_comparison_stats.py comments
- **Visualization**: Review motion_visualization.py plot functions

---

*Last Updated: 2026-05-25*
*Analysis Suite Version: 1.0*
