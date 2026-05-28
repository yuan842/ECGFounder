# System Cleanup Plan: Remove 12-lead + Fine-tuned Models

**Date**: 2026-05-27
**Goal**: Reduce the system to a single model surface (`1_lead_ECGFounder.pth`) with v2 feature-gate FP suppression as the only suppression mechanism. Remove 12-lead, fine-tuned, and v1 LR suppressor code paths.

---

## 1. Checkpoint Files

| File | Size | Action |
|---|---|---|
| `checkpoint/1_lead_ECGFounder.pth` | 124MB | **Keep** |
| `checkpoint/12_lead_ECGFounder.pth` | 370MB | Delete |
| `checkpoint/finetuned_afib_model.pth` | 124MB | Delete |
| `checkpoint/finetuned_ambulatory_model.pth` | 124MB | Delete |
| `checkpoint/finetuned_afib_fuzzy_model.pth` | (if exists) | Delete |
| `checkpoint/afib_fp_suppressor_lr.npz` | 4KB | Delete |
| `checkpoint/perclass_fp_suppressor_Isolated_Supraventricular_Beat_lr.npz` | 4KB | Delete |
| `checkpoint/perclass_fp_suppressor_Isolated_Ventricular_Beat_lr.npz` | 4KB | Delete |

**Disk recovered**: ~742MB

---

## 2. Scripts to Delete Entirely

### 2a. Comparison / fine-tuned model scripts (16 files)
| File | Purpose | Tracked? |
|---|---|---|
| `compare_afib_models.py` | Compare base vs finetuned AFib | tracked |
| `compare_all_models.py` | Multi-model comparison | tracked |
| `compare_mitdb_models.py` | MITDB across model variants | tracked |
| `compare_models_on_fp_dataset.py` | FP dataset model comparison | untracked |
| `compare_150class_ptbxl.py` | PTB-XL across finetuned models | tracked |
| `finetune_afib.py` | Fine-tune AFib head | tracked |
| `finetune_afib_fuzzy.py` | Fine-tune AFib with fuzzy labels | tracked |
| `finetune_model.py` | Generic fine-tune wrapper | tracked |
| `eval_ambulatory_model.py` | Eval ambulatory fine-tune | untracked |
| `eval_compare_models_tprex.py` | TP eval, multi-model | untracked |
| `eval_models_tprex_fixed.py` | TP eval, multi-model | untracked |
| `eval_comprehensive_tprex.py` | TP eval, multi-model | untracked |
| `generate_fuzzy_lead2_ptbxl.py` | Generate fuzzy training data | tracked |
| `create_single_lead_checkpoint.py` | Build 1-lead from 12-lead | tracked |
| `afib_deploy.py` | Deploy script for finetuned AFib | untracked |
| `eval_fuzzylead2.py` | Eval fuzzy-finetuned variant | tracked |

### 2b. v1 LR FP suppressor scripts (12 files)
| File | Purpose | Tracked? |
|---|---|---|
| `train_afib_suppressor.py` | Train AFib LR suppressor | tracked |
| `train_perclass_suppressor.py` | Train per-class LR suppressors | tracked |
| `extract_perclass_fp_features.py` | Feature extraction for v1 | tracked |
| `extract_perclass_tp_features.py` | Feature extraction for v1 | tracked |
| `eval_perclass_suppressors.py` | Eval v1 per-class | tracked |
| `eval_perclass_full.py` | Eval v1 full pipeline | tracked |
| `recalibrate_suppressors.py` | Recalibrate v1 thresholds | tracked |
| `perclass_fp_suppression.py` | v1 per-class module | tracked |
| `afib_fp_suppression.py` | v1 AFib module (only on v2 branch) | (n/a) |
| `afib_optimization_v2.py` | v1 LR tuning | untracked |
| `afib_optimization_v3.py` | v1 LR tuning | untracked |
| `afib_multimetric_optimization.py` | v1 LR tuning | untracked |

---

## 3. Scripts to Modify (Convert to 1-lead / Strip Fine-tuned Refs)

### 3a. PTB-XL / MIT-BIH: convert to 1-lead only
| File | Change |
|---|---|
| `ptbxl_eval.py` | Replace 12-lead checkpoint + adapter with 1-lead Lead II extraction |
| `ptbxl_eval_subset.py` | Same conversion |
| `mitdb_eval.py` | Switch to 1-lead, MLII as input lead |
| `mitdb_eval_standard.py` | Switch to 1-lead, MLII as input lead |
| `mitdb_eval_singlelead.py` | Strip the 12-lead comparison branch, keep 1-lead path |
| `ptbxl_eval_lead_ii.py` | Already 1-lead — minimal change (drop fine-tuned refs if any) |
| `standardize_external_dataset.py` | Drop 12-lead UniversalECGAdapter branches |
| `standardize_external_dataset_robust.py` | Same |
| `standardize_mitdb.py` | Verify 1-lead path only |
| `dataset.py` | Simplify lead config, remove 12-lead branches |
| `fast_download.py` | Only download 1-lead checkpoint |
| `split_mitdb.py` | Verify, likely no change |

### 3b. Fzark/FP eval scripts: strip fine-tuned + v1 LR references
| File | Change | Tracked? |
|---|---|---|
| `eval_ecg_tprex.py` | Remove `AFibFPSuppressor` import & usage | tracked |
| `eval_doctor_removed.py` | Remove `CKPT_FT`, `CKPT_FUZZY`, model loop | untracked |
| `compare_all_classes_on_fp_dataset.py` | Remove finetuned model loop, remove LR suppressor | untracked |
| `compare_tp_fp_motion.py` | Verify no finetuned refs | untracked |
| `optimize_motion_thresholds.py` | Verify no finetuned refs | untracked |

### 3c. Bring `device_utils.py` from v2 branch
The module is imported by 22 scripts but doesn't exist on master. Cherry-pick from `v2-fp-suppression-clean`.

---

## 4. What Remains (the "simplified system")

**Core model + suppression:**
- `net1d.py` — 150-class Net1D architecture
- `checkpoint/1_lead_ECGFounder.pth` — only model
- `device_utils.py` — device resolution helper
- `util.py` — shared utilities

**Data pipeline:**
- `dataset.py` (simplified)
- `standardize_external_dataset.py` (1-lead only)
- `standardize_external_dataset_robust.py` (1-lead only)
- `split_dataset.py`, `split_ecg_tprex.py`, `split_mitdb.py`
- `fast_download.py` (1-lead only)
- `fast_download_ptbxl.py`

**Evaluation:**
- `eval_ecg_tprex.py` — fzark TP eval (with v2 suppression)
- `eval_doctor_removed.py` — fzark FP eval (with v2 suppression)
- `ptbxl_eval.py` / `ptbxl_eval_lead_ii.py` / `ptbxl_eval_subset.py` (1-lead)
- `mitdb_eval.py` / `mitdb_eval_standard.py` / `mitdb_eval_singlelead.py` (1-lead)

**v2 Feature-gate FP suppression (from v2 branch):**
- `multiclass_fp_suppression.py`
- `compare_tp_fzark_with_full_suppression.py`
- `compare_all_classes_with_full_suppression.py`

**Motion / feature analysis:**
- `motion_analysis_utilities.py`, `motion_comparison_stats.py`, `motion_visualization.py`, `motion_analysis_report.py`, `comprehensive_motion_analysis.py`
- `compare_tp_fp_motion.py`, `compare_tp_fp_sqi_motion.py`
- `ecg_feature_analysis.py`, `ecg_filter_design.py`
- `explore_afib_motion.py`, `optimize_motion_thresholds.py`
- `eval_combo_aligned.py`, `evaluate_combo_filter.py`

**Misc:**
- `list_classes.py`, `list_total_classes.py`
- `plot_examples.py`, `visualize_db.py`, `explore_ptbxl_db.py`
- `system_performance.py`
- `tasks.txt`

---

## 5. Execution Order

1. **Branch off master** as `cleanup-12lead-finetune-v1`
2. **Bring over from v2 branch**: `device_utils.py`, `multiclass_fp_suppression.py`, the two `*_with_full_suppression.py` scripts (so the simplified system is functional after cleanup)
3. **Delete** the 36 scripts + 6 checkpoint files listed above
4. **Modify** the PTB-XL / MIT-BIH / dataset scripts to 1-lead-only
5. **Modify** `eval_ecg_tprex.py` etc. to drop v1 LR references
6. **Verify** with `python -c "import <module>"` smoke tests on the keepers
7. **Commit** in logical chunks: (a) delete cruft, (b) bring in v2 essentials, (c) convert to 1-lead

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| 12-lead model still needed for some upstream PTB-XL benchmark | User confirmed: 1-lead Lead II is acceptable |
| Fine-tuned model has institutional value | If needed later, retrievable from git history (`5ade4384ef`) |
| `device_utils.py` cherry-pick introduces incompatible deps | The module is 137 lines, pure-Python torch device logic — should be safe |
| Untracked files contain unsaved analysis | Will list before deletion for sanity check |
