# Lesson: synthetic noise augmentation does not bridge the fzark gap (FROZEN)

**Date**: 2026-05-30
**Status**: ⏸ **FROZEN / parked.** The module `ecg_noise_aug.py` is kept but **not used** —
`--noise-aug` stays opt-in and OFF by default. Do not enable it in training/production now.
**Evidence**: [res/finetune_6head_v2/NOISE_AUG_AB_FZARK.md](../res/finetune_6head_v2/NOISE_AUG_AB_FZARK.md).

## What we tried
Hypothesis: the clean-train→noisy-deploy gap (scope heads strong on clean PTB-XL, weak on
noisy fzark single-lead) could be closed by mixing realistic noise into the clean PTB-XL
training data. Built `ecg_noise_aug.py` (baseline wander / EMG / electrode-motion / powerline /
white, SNR-controlled), wired an opt-in `--noise-aug` into the scope probe, trained a
noise-augmented variant, and A/B'd clean-probe vs noise-probe on the **noisy fzark cohort**.

## Result: no effect
**Mean Δ AUROC (noise − clean) on fzark = +0.001** — within sampling noise on every head
(AFib +0.003, Brady −0.001, SV-Run −0.018 [n=26], V-Run +0.021). Clean PTB-XL fold-10 was
unchanged (0.9697 vs 0.9702), so augmentation neither helped on noisy data nor hurt clean data.

## The lessons (why it's worth recording)
1. **Augmentation can't bridge a hardware/distribution gap.** fzark differs from PTB-XL mostly
   by **device + lead** (128 Hz single-lead ambulatory, different electrode) — a morphology shift,
   not recoverable additive noise. Synthetic baseline/EMG/powerline noise targets the wrong gap.
2. **Where you inject noise matters.** We added noise post-normalization on the band-passed fuzzy
   npz; at inference the 0.67–40 Hz band-pass removes much in-band synthetic noise, so the
   train-time corruption never matched fzark's real residual artifact. Raw-stage (pre-band-pass)
   injection is the correct placement and was never tested.
3. **A frozen-backbone linear probe has little room to gain robustness** — only 6 readout rows
   train; the features are fixed. Robustness would need a full fine-tune.
4. **Validate robustness on the REAL noisy target, not the clean test.** The clean PTB-XL fold-10
   number was identical with/without aug — relying on it would have *hidden* the null result. The
   fzark A/B is what revealed the truth.
5. **Negative results are cheap insurance.** One A/B prevented shipping a "noise-robustness"
   feature that does nothing for the actual deploy domain. The module is *correct* (SNR exact to
   <0.01 dB); the finding is about what gap it can close, not a bug.

## Bonus finding (also recorded)
The PTB-XL fine-tune itself transfers **unevenly** to fzark — helps Bradycardia (+0.005) and
rescues V-Run (0.36→0.45) but slightly hurts AFib (0.890→0.877). That's the device domain gap
showing directly, independent of noise.

## Conditions to revisit (do NOT implement now)
Unfreeze only with a concrete hypothesis addressing the above:
- **Raw-stage noise injection** in the data build (pre-band-pass) — the cheapest untested fix.
- **Full fine-tune** (not linear probe) so the backbone can adapt to ambulatory morphology.
- **Train on real noisy data** (fzark TP) — the true distribution, if labels can be gathered.
- **Direct device adaptation** (fzark calibration / domain adaptation / DualHead routing) — the
  right tool for a hardware gap; augmentation is not.

## What stays in the repo (frozen, not deleted)
- `ecg_noise_aug.py` — frozen module (banner at top), fully tested, importable.
- `scripts/finetune_scope_linprobe.py --noise-aug` — opt-in flag, OFF by default.
- The A/B harness `scripts/eval_probe_on_fzark.py` + report — reusable for any future attempt.
