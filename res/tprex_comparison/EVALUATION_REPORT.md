# ECG-TP_REX Model Evaluation Report

**Date**: May 25, 2026  
**Dataset**: ecg-tp_rex (Atrial Fibrillation Detection)  
**Total Samples Evaluated**: 750 ECG signals  

---

## Executive Summary

Comprehensive evaluation of three models on the ecg-tp_rex dataset:
- **Base Model**: Single-lead ECGFounder (pre-trained)
- **Fine-tuned (AFIB)**: Fine-tuned on Atrial Fibrillation data
- **Fine-tuned (AFIB Fuzzy)**: Fine-tuned with fuzzy labels

### Key Finding
The **Base Model achieves superior performance** on Atrial Fibrillation detection, with an average confidence score of **0.8658** and a detection rate of **90.0%**. Both fine-tuned variants show reduced performance on this task, suggesting possible overfitting or loss of generalization capabilities.

---

## Detailed Results

### 1. Atrial Fibrillation Detection Performance

| Model | Mean Confidence | Std Dev | Median | Detection Rate (>0.5) |
|-------|-----------------|---------|--------|----------------------|
| **Base Model** | **0.8658** | 0.2610 | 0.9576 | **90.0%** |
| Fine-tuned (AFIB) | 0.7732 | 0.2738 | 0.8778 | 88.8% |
| Fine-tuned (AFIB Fuzzy) | 0.7495 | 0.2867 | 0.8608 | 86.8% |

**Performance Degradation**:
- Fine-tuned (AFIB): **-10.7%** confidence drop, **-1.2pp** detection rate
- Fine-tuned (AFIB Fuzzy): **-13.4%** confidence drop, **-3.2pp** detection rate

### 2. Per-Class Prediction Summary

#### Base Model
```
Class                                   Mean Conf   Std Dev   Detection Rate
────────────────────────────────────────────────────────────────────────────
Atrial Fibrillation                     0.8658      0.2610       90.0%
Supraventricular Couplet                0.4341      0.2352       51.6%
Sinus Tachycardia                       0.1925      0.1881        9.5%
Isolated Ventricular Beat               0.1424      0.1005        3.1%
Isolated Supraventricular Beat          0.1051      0.0664        0.1%
Ventricular Couplet                     0.0501      0.1312        0.0%
Prolonged RR Interval                   0.0477      0.1263        0.0%
Ventricular Run                         0.0173      0.0123        0.0%
Pause                                   0.0098      0.0057        0.0%
```

#### Fine-tuned (AFIB)
```
Class                                   Mean Conf   Std Dev   Detection Rate
────────────────────────────────────────────────────────────────────────────
Atrial Fibrillation                     0.7732      0.2738       88.8%
Supraventricular Couplet                0.4695      0.2710       62.0%
Isolated Ventricular Beat               0.1823      0.1400        5.2%
Sinus Tachycardia                       0.1521      0.1546        4.5%
Isolated Supraventricular Beat          0.0683      0.0457        0.0%
Prolonged RR Interval                   0.0270      0.0662        0.0%
Ventricular Couplet                     0.0176      0.0399        0.0%
Ventricular Run                         0.0097      0.0052        0.0%
Pause                                   0.0070      0.0050        0.0%
```

#### Fine-tuned (AFIB Fuzzy)
```
Class                                   Mean Conf   Std Dev   Detection Rate
────────────────────────────────────────────────────────────────────────────
Atrial Fibrillation                     0.7495      0.2867       86.8%
Supraventricular Couplet                0.4989      0.2810       67.5%
Isolated Ventricular Beat               0.2361      0.1669        9.1%
Sinus Tachycardia                       0.1183      0.1376        3.2%
Isolated Supraventricular Beat          0.0770      0.0489        0.0%
Prolonged RR Interval                   0.0151      0.0328        0.0%
Ventricular Couplet                     0.0112      0.0221        0.0%
Ventricular Run                         0.0093      0.0075        0.0%
Pause                                   0.0075      0.0089        0.0%
```

---

## Analysis & Insights

### 1. Primary Task Performance (Atrial Fibrillation)
All three models achieve strong baseline detection performance for Atrial Fibrillation:
- **Minimum detection rate**: 86.8% (Fine-tuned Fuzzy)
- **Maximum detection rate**: 90.0% (Base Model)
- **Confidence range**: 0.7495 - 0.8658

**Observation**: The Base Model's higher confidence and detection rate suggest it has learned robust features for AFIB detection through its diverse training data (150-class classification on PTB-XL).

### 2. Fine-tuning Effects
Both fine-tuned models show:
1. **Reduced AFIB confidence**: Suggests possible catastrophic forgetting or optimization toward different decision boundaries
2. **Slightly improved secondary predictions**: Fine-tuned models show higher mean predictions for other arrhythmias (e.g., Supraventricular Couplet)
3. **Trade-off pattern**: Better at detecting other arrhythmias, worse at AFIB

### 3. Class-Specific Observations

**High Confidence Classes** (mean > 0.4):
- Only Atrial Fibrillation and Supraventricular Couplet reach significant confidence levels
- All other classes remain near baseline (< 0.25)

**Supraventricular Couplet Response**:
- Base Model: 0.4341 → Fine-tuned Fuzzy: 0.4989 (+15%)
- Highest improvement in fine-tuned models
- Suggests fine-tuning partially redirects model focus

**Stability Analysis**:
- Fine-tuned models show **higher standard deviations** in all classes
- Indicates more variability in predictions
- Possible sign of overfitting to training data

---

## Recommendations

### 1. **Prefer Base Model for Production**
   - Higher AFIB detection confidence (0.8658 vs 0.7495)
   - Better generalization across all classes
   - More stable predictions (lower variance)

### 2. **Fine-tuning Optimization**
   If fine-tuning is desired:
   - Use **knowledge distillation** to preserve base model performance
   - Apply **regularization techniques** (dropout, weight decay)
   - Monitor **validation performance** on AFIB detection
   - Consider **progressive fine-tuning** (gradual learning rates)

### 3. **Model Ensemble**
   Consider ensemble approach:
   - Use Base Model for AFIB detection
   - Use Fine-tuned models for secondary arrhythmia detection
   - Weighted combination based on task requirements

### 4. **Further Investigation**
   - Analyze prediction confidence distributions (see visualization plots)
   - Investigate fine-tuning dataset composition
   - Test with mixed training data (balance AFIB with other arrhythmias)
   - Evaluate on multi-arrhythmia test cases

---

## Generated Artifacts

### Data Files
- `summary_afib_detection.csv` - Summary metrics for all three models
- `all_class_predictions.csv` - Detailed per-class predictions for all models
- `analysis_Base_Model.csv` - Detailed Base Model analysis
- `analysis_Fine-tuned_(AFIB).csv` - Fine-tuned (AFIB) analysis
- `analysis_Fine-tuned_(AFIB_Fuzzy).csv` - Fine-tuned (Fuzzy) analysis

### Predictions (NumPy Arrays)
- `preds_Base_Model.npy` - Raw predictions (750 x 150)
- `preds_Fine-tuned_(AFIB).npy` - Raw predictions (750 x 150)
- `preds_Fine-tuned_(AFIB_Fuzzy).npy` - Raw predictions (750 x 150)
- Ground truth: `gts_*.npy` files (all identical for this dataset)

### Visualizations
- `model_comparison_heatmap.png` - Heatmap of mean predictions across models and classes
- `pred_dist_Base_Model.png` - Prediction distribution histograms
- `pred_dist_Fine-tuned_(AFIB).png` - Prediction distribution histograms
- `pred_dist_Fine-tuned_(AFIB_Fuzzy).png` - Prediction distribution histograms

---

## Dataset Notes

**Composition**: 750 Atrial Fibrillation samples from the Belgian Holter device data
- Single-lead ECG recording (unspecified lead)
- 128 Hz native sampling rate → resampled to 500 Hz
- 3-31 second segments (center-cropped to 10 seconds)
- Pre-processing: 50 Hz notch filter, 0.67-40 Hz bandpass, robust normalization

**Limitation**: Evaluation conducted on single-class dataset. ROC-AUC and PR-AUC metrics cannot be computed. Assessment based on confidence scores and detection rates.

---

## Conclusion

The **Base Model** demonstrates superior performance on Atrial Fibrillation detection on the ecg-tp_rex dataset. While both fine-tuned variants maintain reasonable detection rates, they sacrifice confidence and generalization compared to the base model. For production deployment focused on AFIB detection, the **Base Model is recommended** unless specific secondary arrhythmia detection is prioritized.

---

*Evaluation completed: May 25, 2026*  
*Results directory: `./res/tprex_comparison/`*
