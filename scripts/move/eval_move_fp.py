"""MOVE model-performance eval — scope: chest ECG-gel + chest accelerometer.

Compares the BASE backbone vs the production DualHeadECGFounder, with the v2
FP-suppression algorithm OFF and ON.

MOVE has NO arrhythmia labels and the subjects are healthy → the cohort is
treated as all rhythm-negative (like ecg_fp_doctor). The model metric is
therefore the per-V3.1-head FALSE-POSITIVE RATE (alert fires ≥0.5 on a
presumed-negative window; lower = better), broken down by activity phase.

Model input requires preprocessing (the segmentation step saved RAW windows);
inference preprocessing = 50 Hz notch + 0.67–40 Hz bandpass + median baseline +
winsorized z-score (ECGPreprocessor, MOVE = EU 50 Hz, native 500 Hz).

FP algo (replicates multiclass_fp_suppression.ACTIVE_RULES, applied per head)
using the gravity-removed chest motion (production-comparable) + window HR/SNR:
  AFib (idx 5)         : suppress alert if mean_motion > 5.0 mG
  Bradycardia (idx 4)  : suppress alert if mean_hr   > 56.3 bpm
  SV-Trig (idx 16)     : suppress alert if mean_motion < 15.0 mG   (inverted gate)
All other heads pass through unchanged (no rule) → identical off/on.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from preprocessing import ECGPreprocessor
from label_config import FZARK_ONTOLOGY
from sqi import compute_sqi

OUT = "res/move_segmented"
RESULT_DIR = "res/move_eval"
FS = 500

# V3.1 arrhythmia heads (event → idx) + the 8 PTB-XL-specific heads DualHead routes to fuzzy
V31 = {name: node.ecgfounder_index for name, node in FZARK_ONTOLOGY.items()}
PTBXL_SPECIFIC = {2: "NORMAL ECG", 18: "INCOMPLETE RBBB", 26: "LVH", 32: "ATRIAL FLUTTER",
                  36: "LAFB", 62: "INCOMPLETE LBBB", 70: "LPFB", 82: "RVH"}


def fp_gate_keep(event, idx, prob, motion_mg, hr_bpm):
    """Return True if a fired alert is KEPT under the v2 FP algo, False if suppressed.
    Heads without a rule are always kept."""
    if prob < 0.5:
        return False                       # not an alert
    if event == "Atrial Fibrillation":
        return motion_mg <= 5.0
    if event == "Bradycardia":
        return (hr_bpm <= 56.3) if not np.isnan(hr_bpm) else True
    if event == "Supraventricular Trigeminy":
        return motion_mg >= 15.0
    return True                            # no rule → kept


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    df = pd.read_csv(f"{OUT}/move_window_metadata.csv")
    clean = df[df["ecg_pad_fraction"] == 0].reset_index(drop=True)
    print(f"MOVE chest ECG-gel windows: {len(clean)} non-padded (all presumed rhythm-negative)")

    device = resolve_device()
    prep = ECGPreprocessor(powerline_hz=50, normalize="winsorize")   # MOVE = EU 50 Hz, 500 Hz native

    # Preprocess all windows + collect motion (metadata) and HR (sqi)
    print("Preprocessing raw windows + computing motion/HR ...")
    X, motion, hr = [], [], []
    for _, r in clean.iterrows():
        raw = np.load(os.path.join(OUT, r["ecg_path"])).astype(np.float64)
        X.append(prep.process(raw, fs_in=FS))                  # (1,5000) preprocessed
        motion.append(r["chest_motion_mg"])
        acc = np.load(os.path.join(OUT, r["acc_chest_path"]))
        s = compute_sqi(raw, fs=FS, acc_xyz=acc)
        hr.append(s.get("mean_hr_bpm", np.nan))
    X = torch.stack(X).to(device)                              # (N,1,5000)
    motion = np.array(motion, float); hr = np.array(hr, float)
    print(f"  input tensor: {tuple(X.shape)}")

    # Inference — base + DualHead
    def infer(model):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(torch.sigmoid(model(X[i:i+128])).cpu().numpy())
        return np.concatenate(out)
    print("Running BASE ...")
    base = infer(load_ecgfounder(device))
    print("Running DualHead ...")
    dual = infer(DualHeadECGFounder(device, routing="ptbxl_specific"))

    np.save(f"{RESULT_DIR}/base_probs.npy", base)
    np.save(f"{RESULT_DIR}/dual_probs.npy", dual)

    acts = clean["activity_label"].values
    N = len(clean)

    # ── 1. base vs DualHead identity check on V3.1 arrhythmia heads ──────────
    v31_idx = sorted(set(V31.values()))
    max_diff = np.abs(base[:, v31_idx] - dual[:, v31_idx]).max()
    print(f"\n═══ base vs DualHead on V3.1 arrhythmia heads: max |Δ prob| = {max_diff:.2e} "
          f"({'IDENTICAL' if max_diff < 1e-6 else 'DIFFER'}) ═══")
    pt_idx = sorted(PTBXL_SPECIFIC)
    pt_diff = np.abs(base[:, pt_idx] - dual[:, pt_idx]).max()
    print(f"    PTB-XL-specific heads (routed to fuzzy): max |Δ| = {pt_diff:.2e} "
          f"({'differ — DualHead active here' if pt_diff>1e-6 else 'identical'})")

    # ── 2. per-V3.1-head FP rate, FP algo OFF vs ON (base model) ─────────────
    print("\n═══ Per-V3.1-head FALSE-POSITIVE rate on MOVE (base model) — lower is better ═══")
    print(f"  {'event':<32}{'idx':>4}  {'FP off':>8} {'FP on':>8}  {'Δ':>8}  rule")
    rows = []
    for event, idx in sorted(V31.items(), key=lambda kv: kv[1]):
        p = base[:, idx]
        fired = p >= 0.5
        fp_off = float(fired.mean())
        kept = np.array([fp_gate_keep(event, idx, p[i], motion[i], hr[i]) for i in range(N)])
        fp_on = float(kept.mean())
        rule = {"Atrial Fibrillation": "motion≤5",
                "Bradycardia": "HR≤56.3",
                "Supraventricular Trigeminy": "motion≥15"}.get(event, "—")
        print(f"  {event:<32}{idx:>4}  {100*fp_off:>7.1f}% {100*fp_on:>7.1f}%  {100*(fp_on-fp_off):>+7.1f}%  {rule}")
        rows.append(dict(event=event, idx=idx, fp_off_pct=100*fp_off, fp_on_pct=100*fp_on, rule=rule))
    pd.DataFrame(rows).to_csv(f"{RESULT_DIR}/move_fp_by_head.csv", index=False)

    # ── 3. FP rate by ACTIVITY for the motion-gated heads (the MOVE point) ───
    print("\n═══ FP rate by activity — AFib(5), Bradycardia(4), SV-Trig(16) — FP algo OFF→ON ═══")
    order = ["baseline","lift","greetings","gesticulate","walk_before","run","walk_after","unknown"]
    for event, idx in [("Atrial Fibrillation",5), ("Bradycardia",4), ("Supraventricular Trigeminy",16)]:
        print(f"\n  {event} (idx {idx}):")
        print(f"    {'activity':<14}{'n':>5} {'motion_mg':>10} {'FP off':>8} {'FP on':>8}")
        for a in [o for o in order if o in set(acts)]:
            m = acts == a
            p = base[m, idx]; fired = p >= 0.5
            kept = np.array([fp_gate_keep(event, idx, base[j, idx], motion[j], hr[j])
                             for j in np.where(m)[0]])
            print(f"    {a:<14}{m.sum():>5} {np.nanmean(motion[m]):>10.2f} "
                  f"{100*fired.mean():>7.1f}% {100*kept.mean():>7.1f}%")

    print(f"\nArtifacts → {RESULT_DIR}/ (base_probs.npy, dual_probs.npy, move_fp_by_head.csv)")


if __name__ == "__main__":
    main()
