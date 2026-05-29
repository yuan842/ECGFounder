"""Calibrate an SQI gate for Bradycardia (head 4) and Pause (head 142).

From the TP/FP characterization (res/sqi_motion_tp_fp/), these two events separate
on SIGNAL QUALITY, not motion:
  • Bradycardia : false alerts have low SNR / high HF-noise
  • Pause       : false alerts are HF-noise artifacts (the strongest single axis)

We calibrate, per event, two candidate quality gates on the already-computed
per-window SQI panel (no detection, no inference):
  keep iff snr_proxy >= T        (suppress low-SNR)
  keep iff hf_noise  <= T        (suppress high HF-noise)
using the package's supervised Youden-J fitter. TP windows = keep, FP = drop.

Reports thresholds + TP-retained / FP-removed, and the combined (AND) gate.
Writes recommended thresholds to res/sqi_gate_cal/.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd

from fp_suppression import fit_gate_supervised
from fp_suppression.gates import apply_gates, Gate

OUT = "res/sqi_gate_cal"
PW = "res/sqi_motion_tp_fp/per_window.csv"
EVENTS = ["Bradycardia", "Pause"]
# per_window.csv column -> gate feature name (gate framework uses 'hf_noise')
COLMAP = {"snr_proxy": "snr_proxy", "hf_noise_ratio": "hf_noise"}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(PW)
    rows = []
    for ev in EVENTS:
        sub = df[df.event == ev]
        tp = sub[sub.cohort == "TP"]; fp = sub[sub.cohort == "FP"]
        is_tp = np.r_[np.ones(len(tp)), np.zeros(len(fp))].astype(bool)
        print(f"\n══ {ev}  (TP n={len(tp)}, FP n={len(fp)}) ══")
        fitted = {}
        for col, (feat, op) in [("snr_proxy", ("snr_proxy", ">=")),
                                ("hf_noise_ratio", ("hf_noise", "<="))]:
            vals = np.r_[tp[col].to_numpy(float), fp[col].to_numpy(float)]
            try:
                gate, st = fit_gate_supervised(vals, is_tp, feat, op)
            except Exception as e:
                print(f"  {feat:<10} fit failed: {e}"); continue
            fitted[feat] = gate
            print(f"  {feat:<10} {op} {gate.threshold:.4g}  → TP-kept {st['tp_retained']:.2f}  "
                  f"FP-removed {st['fp_removed']:.2f}  J={st['youden_j']:.3f}")
            rows.append(dict(event=ev, feature=feat, op=op, threshold=round(gate.threshold, 5),
                             tp_retained=round(st['tp_retained'], 3),
                             fp_removed=round(st['fp_removed'], 3),
                             youden_j=round(st['youden_j'], 3), n_tp=len(tp), n_fp=len(fp)))
        # combined AND gate (keep iff BOTH quality gates pass)
        if len(fitted) == 2:
            gates = list(fitted.values())
            feats_tp = [{g.feature: r[c] for c, g in zip(["snr_proxy", "hf_noise_ratio"], gates)}
                        for _, r in tp.iterrows()]
            feats_fp = [{g.feature: r[c] for c, g in zip(["snr_proxy", "hf_noise_ratio"], gates)}
                        for _, r in fp.iterrows()]
            tp_keep = np.mean([apply_gates(ev, gates, f).keep for f in feats_tp])
            fp_keep = np.mean([apply_gates(ev, gates, f).keep for f in feats_fp])
            print(f"  COMBINED (snr>=T ∧ hf<=T): TP-kept {tp_keep:.2f}  FP-removed {1-fp_keep:.2f}")
            rows.append(dict(event=ev, feature="snr_proxy∧hf_noise", op="AND",
                             threshold=np.nan, tp_retained=round(tp_keep, 3),
                             fp_removed=round(1 - fp_keep, 3), youden_j=np.nan,
                             n_tp=len(tp), n_fp=len(fp)))
    pd.DataFrame(rows).to_csv(f"{OUT}/sqi_gate_thresholds.csv", index=False)
    print(f"\nArtifacts → {OUT}/sqi_gate_thresholds.csv")


if __name__ == "__main__":
    main()
