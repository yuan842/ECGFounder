# Noise-augmentation A/B on noisy fzark — NEGATIVE result

**Date**: 2026-05-30
**Question**: does realistic noise augmentation (ecg_noise_aug, SNR 5–20 dB, p=0.8, train-only)
improve the scope probe on the **noisy fzark ambulatory cohort**?
**Method**: train clean vs noise-augmented scope probe (identical otherwise), evaluate **base /
clean-probe / noise-probe** on fzark TP (event positives) vs fp_doctor (negatives), per scope head.
**Scripts**: [scripts/finetune_scope_linprobe.py](../../scripts/finetune_scope_linprobe.py) (`--noise-aug`),
[scripts/eval_probe_on_fzark.py](../../scripts/eval_probe_on_fzark.py). Data: `fzark_ab_auroc.csv`.

## Result — AUROC by scope head

| event | head | n_pos | n_neg | base | clean-probe | noise-probe | Δ(noise−clean) |
|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 5 | 500 | 500 | 0.890 | 0.877 | 0.880 | +0.003 |
| Bradycardia | 4 | 500 | 500 | 0.974 | 0.978 | 0.977 | −0.001 |
| Supraventricular Run | 93 | 26 | 500 | 0.606 | 0.579 | 0.560 | −0.018 |
| Ventricular Run | 98 | 500 | 500 | 0.360 | 0.448 | 0.470 | +0.021 |
| **mean Δ (noise − clean)** | | | | | | | **+0.001** |

(Sanity: noise-probe's *clean* PTB-XL fold-10 macro ROC = 0.9697 ≈ clean-probe 0.9702 — aug didn't hurt clean perf.)

## Verdict
**Noise augmentation did not improve noisy-fzark performance** (mean Δ +0.001; every per-head Δ
is within sampling noise, ±0.02–0.03 at n=500). The hypothesis is **not supported** here.

## Why it likely failed (the honest read)
1. **fzark's gap is mostly DEVICE/LEAD, not recoverable additive noise.** fzark is 128 Hz
   single-lead ambulatory on a different electrode; the dominant shift is morphology/lead, which
   synthetic baseline/EMG/powerline noise does not model.
2. **Noise was added post-normalization** (on the already-band-passed fuzzy npz), and at inference
   the preprocessor's 0.67–40 Hz band-pass removes much in-band synthetic noise — so train-time
   noise statistics don't match fzark's real residual artifact.
3. **Linear probe = limited capacity.** Only the 6 readout rows are trainable; the frozen backbone
   features can't adapt, so augmentation can only nudge a linear boundary — little room to gain
   robustness.

## Secondary finding — PTB-XL fine-tune doesn't transfer cleanly to fzark either
The clean-probe vs base on fzark is mixed: **helps Bradycardia (+0.005) and V-Run (0.360→0.448,
escaping worse-than-chance) but slightly HURTS AFib (0.890→0.877)**. So PTB-XL-tuned readouts are
slightly miscalibrated for fzark morphology — consistent with a device/lead domain gap, not noise.

## What would actually help (next experiments)
- **Add noise at the RAW stage (pre-band-pass)** in the data build, so it survives the pipeline
  realistically — the current post-normalization placement is the prime suspect.
- **Train on real noisy data** (fzark TP) — labels are scarce but it's the true distribution.
- **Full fine-tune (not linear probe)** so the backbone can adapt features to ambulatory morphology.
- **Address the device gap directly** — fzark-specific calibration / domain adaptation / the DualHead
  routing approach, rather than expecting augmentation to bridge a hardware gap.

## Caveats
- Single seed; SV-Run n_pos=26 (Δ unreliable); AUROC CIs ~±0.02–0.03 at n=500 → the +0.001 mean is
  indistinguishable from zero.
- The module itself is correct (achieved SNR exact to <0.01 dB) — this is a result about *where/how*
  it's applied and *what gap* it can close, not a bug.
