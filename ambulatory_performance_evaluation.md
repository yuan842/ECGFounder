# Ambulatory Fine-Tuned Model: Multi-Threshold Performance Evaluation Results

This document presents the validation and performance evaluation results of the **Ambulatory Fine-Tuned Model** (which incorporates rhythm-specific motion-gating rules as a post-processing false-positive suppression step) compared to the standard **Fine-Tuned Deep Learning Model** across three different probability thresholds ($0.5$, $0.7$, and $0.75$).

---

## 1. Evaluation Methodology

The model validation was conducted on a representative subset of the **Clinician-Removed False Positives** dataset (`ecg_fp_doctor removed1`). 

### Core Parameters:
* **True Status**: All records evaluated have a verified clinical ground truth of `False` (i.e. every automated rhythm detection is a clinician-confirmed False Positive).
* **Objective**: The goal is to maximize the suppression rate of these false alarms without affecting real arrhythmia sensitivity.
* **Evaluation Thresholds**: Compared at $0.5$, $0.7$, and $0.75$ probability thresholds.
* **Hybrid Architecture**: 
  - **Neural Network Backbone**: Single-lead ECG preprocessed and classified using `finetuned_afib_model.pth` ($150\text{ classes}$, resampled to $500\text{ Hz}$).
  - **Motion-Gating layer**: Aligned 3D-Accelerometer ($5\text{ Hz}$ resampled to $128\text{ Hz}$) dynamically computed Vector Magnitude ($\text{VM}$) to suppress signals with mechanical noise artifacts.

---

## 2. Multi-Threshold Performance Comparison

Below are the comparative results of False Positive (FP) suppression across each major rhythm classification class under different decision thresholds:

### 2.1 Threshold = 0.5 (Standard Clinical Alert)

| Arrhythmia Event | Sample Size | Motion Suppression Rate | Fine-Tuned FP Rate | Ambulatory FP Rate | FP Rate Absolute Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolated Ventricular Beat** | 100 | **100.0%** | 38.0% | 0.0% | **-38.0%** |
| **Sinus Tachycardia** | 100 | **24.0%** | 100.0% | 76.0% | **-24.0%** |
| **Isolated Supraventricular Beat** | 100 | **100.0%** | 8.0% | 0.0% | **-8.0%** |
| **Supraventricular Couplet** | 100 | **100.0%** | 3.0% | 0.0% | **-3.0%** |
| **Atrial Fibrillation (AFib)** | 100 | **0.0%** | 23.0% | 23.0% | 0.0% |

### 2.2 Threshold = 0.7 (Conservative Alert)

| Arrhythmia Event | Sample Size | Motion Suppression Rate | Fine-Tuned FP Rate | Ambulatory FP Rate | FP Rate Absolute Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolated Ventricular Beat** | 100 | **100.0%** | 26.0% | 0.0% | **-26.0%** |
| **Sinus Tachycardia** | 100 | **24.0%** | 98.0% | 75.0% | **-23.0%** |
| **Isolated Supraventricular Beat** | 100 | **100.0%** | 2.0% | 0.0% | **-2.0%** |
| **Supraventricular Couplet** | 100 | **100.0%** | 1.0% | 0.0% | **-1.0%** |
| **Atrial Fibrillation (AFib)** | 100 | **0.0%** | 15.0% | 15.0% | 0.0% |

### 2.3 Threshold = 0.75 (Strict High-Confidence Alert)

| Arrhythmia Event | Sample Size | Motion Suppression Rate | Fine-Tuned FP Rate | Ambulatory FP Rate | FP Rate Absolute Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolated Ventricular Beat** | 100 | **100.0%** | 22.0% | 0.0% | **-22.0%** |
| **Sinus Tachycardia** | 100 | **24.0%** | 98.0% | 75.0% | **-23.0%** |
| **Isolated Supraventricular Beat** | 100 | **100.0%** | 2.0% | 0.0% | **-2.0%** |
| **Supraventricular Couplet** | 100 | **100.0%** | 1.0% | 0.0% | **-1.0%** |
| **Atrial Fibrillation (AFib)** | 100 | **0.0%** | 11.0% | 11.0% | 0.0% |

---

## 3. Analysis of Threshold Impact

1. **Ectopic Beat Classes**:
   - At all three thresholds (0.5, 0.7, 0.75), standard deep learning models still yield false alarms ($38\%$, $26\%$, and $22\%$ respectively for Isolated Ventricular Beats).
   - The **Ambulatory Model** maintains a perfect **0.0% False Positive rate** across all three thresholds. This proves that motion gating acts as an essential, high-confidence fail-safe layer that purely deep learning models cannot reproduce simply by raising their probability threshold.

2. **Sinus Tachycardia**:
   - Raising the deep learning threshold from 0.5 to 0.7 only suppresses $2\%$ of raw false positives (dropping from $100\%$ to $98\%$), showing that ST alarms are highly confident but wrong.
   - The Ambulatory Model successfully suppresses $24\%$ of alerts at 0.5, and $23\%$ of alerts at 0.7/0.75 by identifying physical arm/body movements exceeding $300\text{ mG}$ (e.g. running or rhythmic brushing).

3. **Atrial Fibrillation (AFib)**:
   - Since motion gating was not active for the low-acceleration AFib false alarms, the False Positive rate relies entirely on the deep learning threshold:
     - **0.5 Threshold**: $23.0\%$ False Positives.
     - **0.7 Threshold**: $15.0\%$ False Positives.
     - **0.75 Threshold**: $11.0\%$ False Positives.
   - For AFib, raising the decision threshold is effective at weeding out borderline neural network predictions without needing active motion suppression.
