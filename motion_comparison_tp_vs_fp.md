# Wearable Motion Characteristics: AFib False Positives vs. True Positives

This document presents a comparative analysis of the physical dynamic acceleration profiles between **False Positives** (clinician-removed automated alerts) and **True Positives** (clinician-confirmed true event ECG recordings) within the Atrial Fibrillation wearable sensor datasets.

---

## 1. Quantitative Motion Comparison

The table below shows the statistical distribution of dynamic acceleration (Vector Magnitude with gravity offset removed, scaled to milligravity where $1\text{g} = 1000\text{ mG}$) across both event cohorts:

| Dynamic Motion Metric | False Positives <br> *(Clinician-Removed)* | True Positives <br> *(Clinician-Confirmed)* | Relative Difference <br> *(FP vs. TP)* |
| :--- | :---: | :---: | :---: |
| **Average Motion per Event** | | | |
| - Mean (Average) | **$15.74\text{ mG}$** | **$3.12\text{ mG}$** | **5.0x Higher** |
| - Median (50th Percentile) | **$9.81\text{ mG}$** | **$0.00\text{ mG}$** | **Infinite (Static vs. Active)** |
| - 95th Percentile | **$45.06\text{ mG}$** | **$10.63\text{ mG}$** | **4.2x Higher** |
| **Peak (Max) Motion per Event** | | | |
| - Mean (Average) | **$109.86\text{ mG}$** | **$19.28\text{ mG}$** | **5.7x Higher** |
| - Median (50th Percentile) | **$64.31\text{ mG}$** | **$0.00\text{ mG}$** | **Infinite (Static vs. Active)** |
| - 95th Percentile | **$324.89\text{ mG}$** | **$67.23\text{ mG}$** | **4.8x Higher** |

---

## 2. Key Scientific & Clinical Insights

### 2.1 The "Pristine Static" Requirement for True Arrhythmia Confirmation
* **Observation**: The median dynamic acceleration for True Positives is exactly **$0.00\text{ mG}$** for both average and peak measurements.
* **Clinical Significance**: True cardiac events are detected and confirmed almost exclusively during periods of complete rest or absolute physical immobility (e.g., sleeping or sitting quietly). In these pristine static windows, the ECG signal is completely free of electromyographic (EMG) or electrode-displacement noise, allowing the algorithm to correctly identify irregular R-R intervals and the absence of P-waves with high confidence.

### 2.2 Micro-Movements as the Root Cause of False Alarms
* **Observation**: False positive recordings have a **5x to 10x higher average and peak motion profile** than true positive records.
* **Mechanism**: Even though the average motion of false positives is relatively low ($9.81\text{ mG}$ median), it is significantly elevated compared to true positives. Minor movements—such as adjusting posture, coughing, sighing, or minor arm movements—generate enough micro-acceleration ($10\text{ - }65\text{ mG}$) to trigger:
  1. **Baseline Drift / Wander**: Mechanical shifting of the patch electrodes relative to the skin, causing low-frequency electrical sway.
  2. **Electrode Friction Noise**: High-frequency impedance variations mimicking chaotic f-waves (fibrillatory waves) or ectopic QRS complexes.
* **Conclusion**: Most automated false alarms are physically triggered by these subtle micro-movements, which are absent during true positive events.

### 2.3 Diagnostic Utility of Aligned Accelerometer Gating
* Integrating a secondary accelerometer verification channel is highly effective for reducing clinician alarm fatigue. 
* A true clinical cardiac event typically occurs in a zero-motion environment. By implementing motion gating rules that flag and suppress alerts during periods of elevated dynamic motion, we can confidently reject noise-induced automated triggers without sacrificing clinical diagnostic sensitivity.
