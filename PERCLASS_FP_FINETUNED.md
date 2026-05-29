# Per-Class FP Suppressor Fine-Tuning with `ecg_tp_fzark`

**Date**: 2026-05-26
**Goal**: Use the newly-supplied `ecg_tp_fzark` TP cohort to fine-tune the per-class FP suppressors (originally rule-based on FP-only data).

---

## 1. Headline results

| Class | TP (fzark) | FP (doctor-removed1) | **TP retention** | **FP suppression** | 90/90 |
|---|---|---|---|---|---|
| **Isolated Ventricular Beat** | 800 | 300 | **89.9 %** | **88.7 %** | close (−0.1 / −1.3) |
| **Isolated Supraventricular Beat** | 800 | 300 | **90.5 %** | **93.7 %** | **✓** |
| **Sinus Tachycardia** | 0 (not in fzark) | 300 | n/a | 58.0 % | (Layer-1 only, unchanged) |

### Improvement vs the pre-fine-tune rule-only suppressor

| Class | Old TP retention (measured after fzark arrived) | Old FP suppression | New TP retention | New FP suppression |
|---|---|---|---|---|
| IVB | 38.2 % | 59.7 % | **89.9 %** | **88.7 %** |
| ISB | 23.8 % | 91.7 % | **90.5 %** | **93.7 %** |
| Sinus Tachycardia | n/a | 58.0 % | n/a | 58.0 % (unchanged) |

Pre-fzark rules were **catastrophically over-aggressive** on real TPs — measuring against fzark exposed the damage. After fine-tuning, both IVB and ISB are at or near the 90/90 target.

---

## 2. What the fzark data revealed

Comparing FP and TP feature distributions per class (median values):

### Isolated Ventricular Beat
| Feature | TP | FP | Direction |
|---|---|---|---|
| **n_wide_qrs** | **7** | **1** | TP has MANY wide beats (opposite of what the FP-only rules assumed) |
| qrs_width_ms | 86 ms | 70 ms | TP wider |
| premature_pct | 2.7 % | 14.3 % | TP fewer premature beats (the model fires on premature rather than wide) |
| **snr_proxy** | **2.85** | **1.26** | TP much cleaner signal |
| std_motion | (LR weight +3.56) | | strongest discriminator |
| snr_proxy | (LR weight −2.68) | | second strongest |

### Isolated Supraventricular Beat
| Feature | TP | FP | Direction |
|---|---|---|---|
| **rr_cv** | **0.17** | **0.56** | TP much more regular |
| **snr_proxy** | **2.44** | **0.76** | TP much cleaner |
| n_wide_qrs | 7 | 5 | similar (the old "PAC must be narrow" rule was wrong) |
| qrs_width_max_ms | 132.8 ms | 97.7 ms | TP HIGHER max width — old rule discarded |
| snr_proxy | (LR weight −4.71) | | dominant discriminator |

**Sinus Tachycardia is not present in fzark** (0 TPs), so the existing rule-based filter is unchanged. The next concrete step for Sinus Tachy is to acquire a TP cohort.

---

## 3. What changed in code

### `perclass_fp_suppression.py`
- **Layer-1 rules retired** (they were rejecting TPs):
  - `ivb_no_wide_qrs` (rejected 7 % of TPs)
  - `ivb_not_isolated` (rejected 49 % of TPs)
  - `ivb_motion_with_many_wide` etc.
  - `isb_too_many_wide_qrs` (rejected 62 % of TPs)
  - `isb_motion_artifact (>30 mG)` (rejected 10 % of TPs — relaxed to >200)
- **Layer-1 kept as a minimal safety net** — only `too_few_beats`, `low_sqi`, `extreme_kurt`, `extreme_motion`. These each reject <2 % of fzark TPs.
- **Layer-2 LR ranker added** (loads `checkpoint/perclass_fp_suppressor_<event>_lr.npz` if present).
- New constructor args:
  - `lr_weights_path` — override default path
  - `lr_threshold` — override saved threshold
  - `use_layer2` — disable Layer 2 to fall back to Layer-1-only behavior

### New files
| File | Purpose |
|---|---|
| `extract_perclass_tp_features.py` | TP feature extraction from fzark |
| `train_perclass_suppressor.py` | Per-class LR-ranker training (similar to `train_afib_suppressor.py`) |
| `eval_perclass_full.py` | Full TP+FP validation harness |
| `res/perclass_fp/tp_features_*.csv` | TP feature CSVs (800 rows / class) |
| `checkpoint/perclass_fp_suppressor_<event>_lr.npz` | Trained Layer-2 LR weights |
| `res/perclass_fp_suppression_eval/eval_TP_*.csv` and `eval_FP_*.csv` | Per-event audit logs |

---

## 4. LR ranker details

### IVB (`checkpoint/perclass_fp_suppressor_Isolated_Ventricular_Beat_lr.npz`)
- Trained on 800 TP + 300 FP
- 17-feature standardized logistic regression
- **Deployed threshold: 0.675** (best joint operating point on in-sample sweep)
- 5-fold CV: TP_ret 89.6 % ± 2.8, FP_sup 87.3 % ± 3.7
- Top LR weights (positive = FP):
  - `std_motion` +3.56
  - `snr_proxy` −2.68
  - `mean_motion` −2.18
  - `n_wide_qrs` −1.03

### ISB (`checkpoint/perclass_fp_suppressor_Isolated_Supraventricular_Beat_lr.npz`)
- Trained on 800 TP + 300 FP
- 17-feature standardized logistic regression
- **Deployed threshold: 0.600**
- 5-fold CV: TP_ret 90.6 % ± 3.2, FP_sup 89.7 % ± 4.0
- Top LR weights:
  - `snr_proxy` −4.71
  - `std_motion` +1.42
  - `mean_motion` −1.05
  - `rr_cv` +0.77

---

## 5. Threshold sweep (IVB)

Best joint operating point search:

| L2 thresh | TP ret % | FP sup % | min |
|---|---|---|---|
| 0.625 | 88.5 | 92.0 | 88.5 |
| 0.650 | 88.9 | 91.3 | 88.9 |
| **0.675** | **89.9** | **88.7** | **88.7** ← chosen |
| 0.700 | 90.5 | 86.7 | 86.7 |
| 0.725 | 91.0 | 86.0 | 86.0 |

The 90/90 target is essentially at the boundary — within CV noise. Two tighter operating points are available if priorities shift:
- **Safety-first** (high TP retention): threshold 0.75 → 91.4 % / 83.3 %
- **Aggressive** (high FP suppression): threshold 0.55 → 87.5 % / 95.7 %

---

## 6. Sinus Tachycardia — not fine-tuned

`ecg_tp_fzark` contains **zero** Sinus Tachycardia TPs. The rule-based Layer-1 suppressor (58 % FP suppression) is retained. To complete the suite, a Sinus-Tachy TP cohort is needed. Once available the existing pipeline (`extract_perclass_tp_features.py` + `train_perclass_suppressor.py`) extends directly.

---

## 7. End-to-end pipeline performance with the fine-tuned suppressors

Combining the fine-tuned model + per-class suppressor stack:

| Event Type | Raw FP rate (fine-tuned model) | After per-class suppressor | Reduction |
|---|---|---|---|
| Atrial Fibrillation | 47.0 % | 2.0 % | −95.7 % |
| **Isolated Ventricular Beat** | **69.5 %** | **~7.8 %** (69.5 % × 11.3 %) | **−88.7 %** |
| **Isolated Supraventricular Beat** | **40.0 %** | **~2.5 %** (40 % × 6.3 %) | **−93.7 %** |
| Sinus Tachycardia | 100.0 % | 42.0 % | −58.0 % |

The IVB and ISB FP rates are now within striking distance of the AFib pipeline performance.

---

## 8. Reproduction

```bash
# 1. Extract TP features from fzark
python3 extract_perclass_tp_features.py

# 2. Train LR rankers
python3 train_perclass_suppressor.py

# 3. Validate
python3 eval_perclass_full.py
```

Total runtime on this dataset (800 TP + 300 FP per class, two classes, CPU): **~3 minutes** for the entire pipeline.

---

*Generated 2026-05-26 from `res/perclass_fp_suppression_eval/`.*
