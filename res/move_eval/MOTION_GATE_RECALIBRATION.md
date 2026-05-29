# Per-Device Motion-Gate Recalibration — MOVE chest-strap

**Date**: 2026-05-29
**Scope**: MOVE chest ECG-gel + chest accelerometer; AFib head (idx 5).
**Prototype**: [scripts/move/recalibrate_motion_gate.py](../../scripts/move/recalibrate_motion_gate.py)
**Upstream**: [scripts/move/eval_move_fp.py](../../scripts/move/eval_move_fp.py) (base vs DualHead, FP off/on).

---

## Problem

The production AFib motion gate suppresses an alert when `mean_motion > 5 mG` (tuned on the fzark ambulatory cohort). On the MOVE chest device (analyzed ECG channel: `ecg:gel`), chest motion is **< 2 mG even during running**, so the 5 mG gate almost never fires and motion-induced AFib false positives survive.

## Why the AFib head is the clean calibration target

Running does not cause true AFib, so **every AFib alert during `run` is a false positive** (motion artifact). The `baseline` (rest) AFib FP rate is the model's irreducible non-motion FP floor. A correctly-tuned gate should pull motion-phase FP down toward that floor without over-suppressing rest windows (where real AFib would appear). This lets us calibrate the FP side label-free.

## Finding 1 — the fzark threshold is off-scale for this device

| MOVE motion (gravity-removed `mean_motion_mg`) | value |
|---|---|
| median | 1.01 mG |
| p95 | 3.69 mG |
| p99 | 5.14 mG |
| max | 23.1 mG |
| **fzark 5 mG threshold sits at** | **the 98.3rd percentile** |

→ The 5 mG gate suppresses only **1.7 %** of MOVE windows. It is effectively inert on this device.

## Finding 2 — rest motion envelope → device threshold

REST (baseline) motion, n=231: p90 = 0.72, p95 = 0.80, p99 = 1.05 mG.

**Proposed device threshold T_dev = p95(rest) = 0.80 mG** — alerts with more motion than a resting subject ever shows are motion-contaminated. That's **6× lower** than the fzark 5 mG.

## Finding 3 — threshold sweep (AFib FP, keep iff motion ≤ T)

| T (mG) | FP overall | FP rest | FP run | run FP removed |
|---|---|---|---|---|
| ∞ (algo OFF) | 13.3% | 2.2% | 16.6% | 0.0 pp |
| 5.00 (fzark prod) | 12.0% | 2.2% | 14.3% | 2.2 pp |
| 2.00 | 8.5% | 2.2% | 8.8% | 7.7 pp |
| 1.50 | 6.4% | 2.2% | 5.6% | 11.0 pp |
| 1.00 | 3.0% | 1.7% | 0.8% | 15.7 pp |
| **0.80 (T_dev)** | **1.7%** | **1.3%** | **0.5%** | **16.0 pp** |
| 0.50 | 0.0% | 0.0% | 0.0% | 16.6 pp |

## Verdict

- **fzark 5 mG removes only 2.2 pp** of run-phase AFib FP (16.6 % → 14.3 %).
- **T_dev = 0.80 mG removes 16.0 pp** (16.6 % → 0.5 %), bringing run FP essentially to the 2.2 % rest floor.
- **Rest-phase FP is preserved** at T_dev: 2.2 % → 1.3 % (a mild over-trim — see the TP-retention caveat).

## Recommended device threshold

| Threshold | run FP | rest FP | Trade-off |
|---|---|---|---|
| **1.0 mG** (≈ p99 rest) | 0.8 % | 1.7 % | **Recommended** — kills nearly all motion FP, barely touches the rest floor → safest for TP retention |
| 0.80 mG (p95 rest) | 0.5 % | 1.3 % | More aggressive; trims ~0.9 pp of rest alerts (potential TP loss) |

**Recommend ≈ 1.0 mG** for the MOVE chest-strap device — it removes ~95 % of the motion-induced AFib FP while leaving the rest-phase rate essentially untouched.

## Scope & caveats

- **This is per-device calibration, NOT a change to the fzark production gate.** The fzark/Vivalink gate stays at 5 mG for its own motion distribution. The MOVE chest device (`ecg:gel`) uses ≈1 mG. The deliverable is the *method* (calibrate the gate to the device's rest-motion envelope) + the MOVE-specific value.
- **TP-retention is not validated here** — MOVE has no labelled positives. The recalibrated threshold trims the FP side and respects the rest envelope, but confirming it doesn't suppress real low-motion arrhythmias requires re-running the threshold against the **fzark TP cohort** (which has clinician-confirmed positives + motion). That is the next step before any deployment.
- **Only the AFib head was calibrated.** The same rest-envelope method applies to the other motion-gated heads (Bradycardia HR-gate is unaffected; SV-Trig's inverted motion≥15 gate already suppresses everything on MOVE and needs its own device-specific re-think).

## Reproduce

```bash
python3 scripts/move/eval_move_fp.py            # produces res/move_eval/base_probs.npy
python3 scripts/move/recalibrate_motion_gate.py # this analysis
```
