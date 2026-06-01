# L1 ScopeProjection — training + evaluation (PTB-XL only)

**What:** the L1 overlay layer ([overlay/scope_overlay.py](../../overlay/scope_overlay.py)) — a learned linear projection mapping the frozen backbone's **150 logits → 6 scope probabilities** `{4 Brady, 5 AFib, 6 Sinus-Tachy, 93 SVT-Run, 98 V-Run, 142 Pause}`. Trainer/eval: [scripts/train_scope_overlay.py](../../scripts/train_scope_overlay.py) (`--ptbxl-only`). Checkpoint: `scope_projection.pth`; metrics: `metrics.csv`.

**Trained on PTB-XL only** — no fuzzy augmentation, no fzark. Single dataset, authors' fold split.

## Setup

- **Backbone:** frozen `1_lead_ECGFounder.pth`; L1 is the only trainable part (906 weights + 6 temperatures).
- **Features:** backbone 150-logits of PTB-XL **lead-II** (from cached `base_probs_full.npy` → logits), 21,799 records. Labels = the 6 scope columns of the authoritative `csv/ptbxl_label.csv` 150-vectors.
- **Split (by ecg_id, patient-stratified):** folds 1-8 train (17,418) / 9 val (2,183) / 10 test (2,198).
- **Loss:** BCEWithLogits, per-head `pos_weight`; early-stop on mean val PR-AUC; per-head temperature fit on val (`[0.7 0.6 0.6 0.7 0.7 1.0]`).

## Results — fold-10 test, BASE backbone head vs L1 projection

| head | n_pos | AUROC base→L1 | PR-AUC base→L1 | PPV@0.5 base→L1 |
|---|---:|---|---|---|
| 4 Bradycardia | 64 | 0.944→**0.960** | 0.346→**0.601** | 0.162→0.168 |
| 5 AFib | 152 | 0.976→**0.982** | 0.909→**0.922** | 0.683→**0.724** |
| 6 Sinus-Tachy | 82 | 0.986→0.980 | 0.813→0.805 | 0.328→**0.583** |
| 93 SVT-Run | 5 | 0.996→0.990 | 0.319→0.163 | 0.200→0.046 |
| 98 V-Run | 5 | 0.997→0.993 | 0.341→0.215 | 0.300→0.060 |
| 142 Pause | 0 | — | — | — |

(sens@0.5 in `metrics.csv`; AFib sens unchanged 0.934, others trade a little recall for precision.)

## Findings

1. **L1 clearly helps the three data-rich heads.**
   - **Bradycardia:** PR-AUC **0.35→0.60** (nearly doubled), AUROC +0.016. The single biggest win — the projection reads the descriptor scaffold (rate/rhythm context) the raw single head ignores.
   - **AFib:** improves on *all three* metrics (AUROC, PR-AUC, and PPV 0.68→0.72) — already the strongest head, still lifted.
   - **Sinus-Tachy:** PPV **0.33→0.58** (recall traded slightly, PR-AUC flat) — far fewer false tachy calls at 0.5.

2. **L1 hurts the two rare heads (SVT-Run, V-Run).** With ~37 train positives each, the projection under-calls them. **Caveat:** only n=5 test positives → very noisy, and AUROC stays ~0.99. Directionally, do not route 93/98 through L1.

3. **Pause (142) untrainable** — 0 PTB-XL positives. It will never fire from an L1 trained on PTB-XL alone.

## Recommendation (per-head routing)

| heads | source | rationale |
|---|---|---|
| 4 Brady, 5 AFib, 6 Sinus-Tachy | **L1 projection** | calibration/precision gains, esp. Brady PR-AUC + Tachy PPV |
| 93 SVT-Run, 98 V-Run | **base head** (then L2 merges) | too few PTB-XL labels; L1 degrades them; entanglement unfixable at single-lead |
| 142 Pause | **base head** (untrained) | 0 PTB-XL positives — needs another dataset before L1 can learn it |

## Caveats

- **AUROC is ~flat everywhere** — the backbone's ranking was already good; L1's win is **calibration / precision at the 0.5 operating point**, not discrimination.
- Features are `logit(base_probs_full)`; lead-II only.
- This is L1 only. L2 ([overlay/arbiter.py](../../overlay/arbiter.py)) applies on top; SQG stays OFF.
- SVT/VT/Pause remain the known gaps; the user's decision was to keep training PTB-XL-only, so those stay on the base head per the routing table above.
