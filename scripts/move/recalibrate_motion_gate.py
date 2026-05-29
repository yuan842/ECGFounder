"""Per-device motion-gate recalibration prototype (MOVE chest-strap dry electrode).

Problem (from eval_move_fp.py): the production AFib motion gate suppresses an
alert when mean_motion > 5 mG. That 5 mG threshold was tuned on the fzark
ambulatory cohort; MOVE chest-strap motion is < 2 mG even during running, so
the gate almost never fires and motion-induced AFib false positives survive.

Calibration target — the AFib head:
  Running does NOT cause true AFib, so every AFib alert during `run` is a false
  positive (motion artifact). The `baseline` (rest) AFib FP rate is the model's
  irreducible non-motion FP floor. A correctly-tuned motion gate should bring
  the motion-phase FP rate down toward that rest floor WITHOUT over-suppressing
  rest windows (which is where real AFib would appear).

Method (label-free, defensible):
  1. Characterize the device motion distribution (gravity-removed mean_motion_mg)
     overall + per activity. Show where the fzark 5 mG threshold sits.
  2. Derive a device threshold from the REST (baseline) motion envelope:
     T_dev = high percentile of baseline-phase motion (alerts with more motion
     than a resting subject ever shows are motion-contaminated → suppress).
  3. Sweep the threshold; report AFib FP rate overall + rest + run for each,
     and identify T* where run-phase FP approaches the rest floor.

Caveat: MOVE has no labelled positives, so TP-retention at T_dev cannot be
measured here — that requires the fzark TP cohort. This prototype recalibrates
the FP side and the rest-envelope; TP validation is the next step.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

OUT = "res/move_segmented"
EVAL = "res/move_eval"
AFIB_IDX = 5
FZARK_T = 5.0          # production threshold (mG), tuned on fzark


def main():
    clean = pd.read_csv(f"{OUT}/move_window_metadata.csv")
    clean = clean[clean.ecg_pad_fraction == 0].reset_index(drop=True)
    base = np.load(f"{EVAL}/base_probs.npy")
    motion = clean["chest_motion_mg"].to_numpy(float)
    acts = clean["activity_label"].to_numpy()
    afib_fired = base[:, AFIB_IDX] >= 0.5

    # ── 1. device motion distribution ───────────────────────────────────────
    print("═══ 1. MOVE chest motion distribution (gravity-removed mean_motion_mg) ═══")
    pct = [50, 90, 95, 99, 99.9, 100]
    overall = np.percentile(motion, pct)
    print(f"  overall percentiles: " + "  ".join(f"p{p}={v:.2f}" for p, v in zip(pct, overall)))
    print(f"  → fzark production threshold {FZARK_T} mG sits at the "
          f"{100*np.mean(motion <= FZARK_T):.2f}th percentile of MOVE motion "
          f"(i.e. it suppresses only {100*np.mean(motion > FZARK_T):.2f}% of windows — barely fires).")

    base_motion = motion[acts == "baseline"]
    print(f"\n  REST (baseline) motion envelope, n={len(base_motion)}:")
    for p in [90, 95, 99]:
        print(f"    p{p} = {np.percentile(base_motion, p):.3f} mG")

    # device threshold = p95 of rest motion (alerts above the resting envelope → suppress)
    T_dev = float(np.percentile(base_motion, 95))
    print(f"\n  → proposed device threshold T_dev = p95(rest motion) = {T_dev:.3f} mG "
          f"(vs fzark {FZARK_T} mG — {FZARK_T/T_dev:.0f}× lower)")

    # ── 2. threshold sweep: AFib FP rate overall / rest / run ────────────────
    print("\n═══ 2. AFib motion-gate sweep — keep alert iff motion ≤ T ═══")
    print(f"  rest(baseline) FP floor = model's irreducible non-motion FP")
    print(f"  {'T (mG)':>8} {'FP overall':>11} {'FP rest':>9} {'FP run':>9} {'run suppressed':>15}")
    rest_mask = acts == "baseline"; run_mask = acts == "run"
    def fp_at(T):
        kept = afib_fired & (motion <= T)
        return (100*kept.mean(),
                100*(kept & rest_mask).sum()/rest_mask.sum(),
                100*(kept & run_mask).sum()/run_mask.sum())
    rest_floor = 100*(afib_fired & rest_mask).sum()/rest_mask.sum()  # = FP rest at T=inf
    for T in [np.inf, FZARK_T, 3.0, 2.0, 1.5, 1.0, T_dev, 0.5]:
        o, r, rn = fp_at(T)
        run_off = 100*(afib_fired & run_mask).sum()/run_mask.sum()
        supp = run_off - rn
        tag = ""
        if T == np.inf: tag = "(FP algo OFF)"
        elif T == FZARK_T: tag = "(fzark prod)"
        elif abs(T - T_dev) < 1e-9: tag = "(T_dev = p95 rest)"
        print(f"  {('inf' if np.isinf(T) else f'{T:.2f}'):>8} {o:>10.1f}% {r:>8.1f}% {rn:>8.1f}% "
              f"{supp:>13.1f}pp  {tag}")

    # ── 3. verdict ───────────────────────────────────────────────────────────
    o_dev, r_dev, rn_dev = fp_at(T_dev)
    o_fz, r_fz, rn_fz = fp_at(FZARK_T)
    run_off = 100*(afib_fired & run_mask).sum()/run_mask.sum()
    print("\n═══ 3. Recalibration verdict (AFib head) ═══")
    print(f"  run-phase AFib FP:  OFF={run_off:.1f}%  |  fzark 5mG={rn_fz:.1f}%  |  T_dev={rn_dev:.1f}%")
    print(f"  rest floor (target): {rest_floor:.1f}%")
    print(f"  rest-phase AFib FP at T_dev: {r_dev:.1f}%  (should stay ≈ floor — confirms rest alerts preserved)")
    print(f"\n  → fzark 5 mG removes {run_off-rn_fz:.1f} pp of run FP; "
          f"T_dev={T_dev:.2f} mG removes {run_off-rn_dev:.1f} pp, "
          f"bringing run FP from {run_off:.1f}% toward the {rest_floor:.1f}% rest floor.")
    print(f"  Device-specific threshold ≈ {T_dev:.2f} mG for MOVE chest-strap "
          f"(NOT a change to the fzark production gate — per-device calibration).")
    print(f"\n  ⚠ TP-retention at T_dev needs the fzark TP cohort (no positives in MOVE).")


if __name__ == "__main__":
    main()
