# Model Comparison on ECG-FP-Doctor-Removed1 Dataset

**Date**: pipeline run

**Cohort**: 500 AFib FP records (clinician-removed) of 1793 total

All records are clinician-removed false positives. The correct system behavior is **no alert**.

## Headline numbers

| Model | Model threshold | Raw alerts | Raw alert rate | Post-suppression alerts | Post alert rate | FP suppression gain |
|---|---|---|---|---|---|---|
| **baseline** | 0.5000 | 347 | 69.4% | 15 | 3.0% | 95.7% |
| **finetuned** | 0.2646 | 238 | 47.6% | 10 | 2.0% | 95.8% |

*Raw alert rate = how often the model alone would flag AFib (lower is better since all are FP).*  
*Post alert rate = remaining alerts after Layer 1 + Layer 2 suppression.*  
*FP suppression gain = fraction of the raw alerts that the suppressor removed.*

## Output files

- `./res/fp_model_comparison/comparison_summary.csv`
- `./res/fp_model_comparison/per_event_predictions.csv`
- `./res/fp_model_comparison/baseline_probs.npy`
- `./res/fp_model_comparison/finetuned_probs.npy`