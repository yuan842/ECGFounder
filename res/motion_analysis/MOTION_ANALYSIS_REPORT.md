# Motion/Accelerometer Analysis Report
## ECG-FP-Doctor-Removed1 vs ECG-TP_REX Comparison

**Date Generated**: 2026-05-26 00:00:02
**Analysis Type**: Cross-dataset motion characteristics comparison

---


## Executive Summary

**Overall Finding**: False Positive (FP) detections show **2.3x higher mean motion** (17.89 mG) compared to True Positive (TP) detections (7.64 mG).


### Key Insights


1. **Statistical Significance**: 6/6 motion metrics show statistically significant differences (p < 0.05)


2. **True Event Characteristics**: 48.1% of TP events occur in pristine static conditions (mean motion < 1 mG), suggesting these are genuine arrhythmias in controlled environments


3. **Effect Sizes**: 1 event types show large effect sizes (Cohen's d > 0.8) between FP and TP, indicating substantial practical differences


### Clinical Implications


- **Motion Gating Effectiveness**: High motion in FP detections suggests motion artifacts or false detections triggered by physical activity rather than true arrhythmias

- **True Event Profile**: TP events predominantly occur during low motion periods, supporting the hypothesis that true arrhythmias are detected during resting states

- **Device Optimization**: Motion-based filtering/gating can effectively reduce false positive detections without compromising true event detection


## Dataset Overview

### Record Counts

| Dataset | Total Records | Feature Extraction Success Rate |
|---------|---------------|--------------------------------|
| **FP (False Positives)** | 500 | 100.0% |
| **TP (True Positives)** | 750 | 100.0% |

### Event Type Distribution

**False Positives (FP):**

| Event Type | Count |
|---|---|
| Supraventricular Trigeminy | 47 |
| Atrial Fibrillation | 46 |
| Isolated Supraventricular Beat | 46 |
| Multiple Event | 45 |
| Ventricular Couplet | 45 |
| Pause | 41 |
| Sinus Tachycardia | 40 |
| Ventricular Run | 38 |
| Supraventricular Run | 36 |
| Bradycardia | 29 |
| Ventricular Trigeminy | 29 |
| Isolated Ventricular Beat | 29 |
| Supraventricular Couplet | 26 |
| Custom Heart Rate | 2 |
| Prolonged RR Interval | 1 |

*Total: 500 FP records analyzed*

**True Positives (TP):**

| Event Type | Count |
|---|---|
| Atrial Fibrillation | 750 |

*Total: 750 TP records analyzed*


## Overall Motion Characteristics

### Global FP vs TP Comparison

| Metric | FP Mean | TP Mean | FP/TP Ratio | Effect Size | p-value | Significance |
|--------|---------|---------|-------------|-------------|---------|---------------|
| mean_motion | 17.89 | 7.64 | 2.34x | 0.88 | 1.84e-29 | *** |
| max_motion | 410.97 | 208.15 | 1.97x | 1.27 | 3.44e-54 | *** |
| std_motion | 52.98 | 25.71 | 2.06x | 1.34 | 6.68e-75 | *** |
| median_motion | 6.73 | 2.72 | 2.48x | 0.54 | 3.73e-22 | *** |
| peak_ratio | 29.99 | 14.97 | 2.00x | 1.13 | 6.53e-72 | *** |
| dom_freq | 0.09 | 0.07 | 1.23x | 0.24 | 1.87e-15 | *** |

**Legend**: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

### Interpretation

- **Mean Motion**: Average dynamic motion magnitude during detected events
- **Max Motion**: Peak motion during the event
- **Std Motion**: Variability of motion during events
- **Median Motion**: Central tendency (robust to outliers)
- **Peak Ratio**: Ratio of max to mean (measures motion consistency)


## Per-Event-Type Analysis

### Atrial Fibrillation

| Metric | FP Mean | TP Mean | FP/TP Ratio | Effect Size (Cohen's d) | p-value |
|--------|---------|---------|-------------|-------------------------|----------|
| mean_motion | 27.38 | 7.64 | 3.58x | 2.01 | 2.59e-16 |
| max_motion | 428.24 | 208.15 | 2.06x | 1.13 | 8.96e-15 |
| std_motion | 57.32 | 25.71 | 2.23x | 1.29 | 1.28e-18 |
| median_motion | 13.00 | 2.72 | 4.78x | 1.93 | 1.02e-14 |
| peak_ratio | 22.40 | 14.97 | 1.50x | 0.50 | 7.09e-04 |


## Motion Patterns & Insights

### Zero-Motion Analysis

- **FP Events with Zero Motion** (<1 mG): 0.0%
- **TP Events with Zero Motion** (<1 mG): 48.1%

**Interpretation**: The significantly higher proportion of TP events in zero-motion conditions indicates that true arrhythmias are predominantly detected during patient rest, while false positives are more commonly associated with physical activity.

### Peak Motion Behavior

- **FP Mean Peak-to-Mean Ratio**: 29.99
- **TP Mean Peak-to-Mean Ratio**: 14.97

**Interpretation**: Peak-to-mean ratio characterizes motion consistency. FP events with higher ratios suggest sporadic motion spikes, while TP events with lower ratios indicate more sustained motion patterns.

### Frequency Characteristics

- **FP Mean Dominant Frequency**: 0.09 Hz
- **TP Mean Dominant Frequency**: 0.07 Hz

**Interpretation**: Dominant frequency reveals the primary motion component. Higher frequencies in FP events may indicate artifacts or rapid movements, while lower frequencies in TP events suggest slower, more sustained physiological processes.


## Methodology

### Motion Feature Extraction

1. **Accelerometer Data Loading**: 3-axis raw acceleration (x, y, z) extracted from JSON records
2. **Normalization**: Raw values divided by scale factor (2048.0) to convert to G units
3. **Vector Magnitude**: |v| = √(x² + y² + z²) computed in G units
4. **DC Removal**: Gravity component removed using 5-sample rolling mean
5. **Unit Conversion**: Dynamic motion converted to milliG (mG = G × 1000)

### Statistical Analysis

- **Descriptive Statistics**: Mean, median, standard deviation, percentiles (25, 50, 75, 90, 95, 99)
- **Parametric Tests**: Independent t-test (assumes normal distribution)
- **Non-Parametric Tests**: Mann-Whitney U test (robust to non-normal distributions)
- **Effect Size**: Cohen's d to quantify practical significance
- **Significance Threshold**: p < 0.05 (α = 0.05)

### Sampling Strategy

- **FP Dataset**: 500 records from original 500 (100.0% of total)
- **TP Dataset**: 750 records from original 750 (all available records)

### Assumptions & Limitations

1. **Accelerometer Data Quality**: Assumes accurate 3-axis acceleration recording at 5 Hz
2. **DC Removal**: Uses rolling mean to approximate gravity; may not account for tilting
3. **Sampling Rate**: 5 Hz accelerometer may miss high-frequency motion components (>2.5 Hz)
4. **Single-Lead ECG**: Analysis does not account for device orientation or axis variations
5. **Population**: Findings specific to Belgian Holter device data; may not generalize to other devices


## Appendices

### A. Detailed Statistics by Dataset

#### False Positive (FP) Dataset

**Mean Motion**: 17.885383200727805

#### True Positive (TP) Dataset

**Mean Motion**: 7.6411046854654945

### B. Validation Report

#### FP Dataset Validation

- Total records processed: 500
- Features extracted: 500
- Data quality: PASS

#### TP Dataset Validation

- Total records processed: 750
- Features extracted: 750
- Data quality: PASS

### C. Comparison Summary Table

| metric        | event_type          |     fp_mean |   fp_median |     tp_mean |   tp_median |   ratio_mean |   ratio_median |   cohens_d |     p_value |   fp_n |   tp_n |
|:--------------|:--------------------|------------:|------------:|------------:|------------:|-------------:|---------------:|-----------:|------------:|-------:|-------:|
| mean_motion   | OVERALL             |  17.8854    |  11.9073    |   7.6411    |  10.8307    |      2.34068 |        1.09941 |   0.883614 | 1.84343e-29 |    500 |    750 |
| max_motion    | OVERALL             | 410.971     | 399.675     | 208.148     | 386.552     |      1.97442 |        1.03395 |   1.27276  | 3.4422e-54  |    500 |    750 |
| std_motion    | OVERALL             |  52.9821    |  49.6882    |  25.711     |  47.7545    |      2.06068 |        1.04049 |   1.34225  | 6.67656e-75 |    500 |    750 |
| median_motion | OVERALL             |   6.73483   |   3.18719   |   2.7184    |   2.58725   |      2.4775  |        1.23188 |   0.540717 | 3.7267e-22  |    500 |    750 |
| peak_ratio    | OVERALL             |  29.9916    |  33.2446    |  14.9681    |  16.8397    |      2.00369 |        1.97418 |   1.12974  | 6.53441e-72 |    500 |    750 |
| dom_freq      | OVERALL             |   0.0892376 |   0.0645161 |   0.0725591 |   0.0322581 |      1.22986 |        2       |   0.240502 | 1.87056e-15 |    500 |    750 |
| mean_motion   | Atrial Fibrillation |  27.3769    |  19.5437    |   7.6411    |  10.8307    |      3.58284 |        1.80447 |   2.01389  | 2.59101e-16 |     46 |    750 |
| max_motion    | Atrial Fibrillation | 428.244     | 404.395     | 208.148     | 386.552     |      2.0574  |        1.04616 |   1.12516  | 8.95947e-15 |     46 |    750 |
| std_motion    | Atrial Fibrillation |  57.3246    |  50.6152    |  25.711     |  47.7545    |      2.22957 |        1.0599  |   1.29255  | 1.2769e-18  |     46 |    750 |
| median_motion | Atrial Fibrillation |  12.996     |   6.26752   |   2.7184    |   2.58725   |      4.78076 |        2.42247 |   1.92616  | 1.02261e-14 |     46 |    750 |
| peak_ratio    | Atrial Fibrillation |  22.4026    |  22.2238    |  14.9681    |  16.8397    |      1.49669 |        1.31973 |   0.502244 | 0.00070935  |     46 |    750 |

---

*Report generated on 2026-05-26 at 00:00:02*
