"""Per-head threshold calibration for the run/pause scope heads (93, 98, 142).

These heads never fire at the default 0.5 on the fzark TP cohort (0% sensitivity).
This sweeps each head's threshold on the matched TP-positive vs fp_doctor-negative
scores (same stratified alignment as combined_binary_eval.py) and reports the
operating points, so we can pick a per-head threshold that recovers recall.

Heads (fzark event → tasks.txt head):
  93  Supraventricular Run  → SUPRAVENTRICULAR TACHYCARDIA
  98  Ventricular Run       → VENTRICULAR TACHYCARDIA
 142  Pause                 → WITH SINUS PAUSE
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

TP_DIR, FP_DIR = "data/ecg_tp_fzark", "data/ecg_fp_doctor removed1"
TP_PROBS = "res/tp_fzark_full_suppression/baseline_probs_tp.npy"
FP_PROBS = "res/fp_allclass_full_suppression/baseline_probs_full.npy"
HEADS = {93: "Supraventricular Run", 98: "Ventricular Run", 142: "Pause"}
OUT = "res/scope_threshold_cal"


def load(summary, probs, seed=42, per_class=500):
    df = pd.read_csv(summary)
    df = pd.concat([g.sample(n=min(per_class, len(g)), random_state=seed)
                    for _, g in df.groupby('Event Type')]).reset_index(drop=True)
    return df, np.load(probs)


def sweep(pos, neg):
    grid = np.unique(np.concatenate([pos, neg, np.linspace(0, 1, 201)]))
    rows = []
    npos, nneg = len(pos), len(neg)
    for t in grid:
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum())
        sens = tp / npos; spec = (nneg - fp) / nneg
        ppv = tp / max(1, tp + fp); f1 = 2 * tp / max(1, 2 * tp + fp + (npos - tp))
        rows.append((t, sens, spec, ppv, f1, sens + spec - 1, tp, fp))
    return pd.DataFrame(rows, columns=["thr", "sens", "spec", "ppv", "f1", "youden", "tp", "fp"])


def main():
    os.makedirs(OUT, exist_ok=True)
    tp_df, tp_p = load(f"{TP_DIR}/summary.csv", TP_PROBS)
    fp_df, fp_p = load(f"{FP_DIR}/summary.csv", FP_PROBS)
    summary = []
    for h, ev in HEADS.items():
        pos = tp_p[(tp_df['Event Type'] == ev).values, h]
        neg = fp_p[(fp_df['Event Type'] == ev).values, h]
        auroc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        sw = sweep(pos, neg); sw.to_csv(f"{OUT}/sweep_head{h}.csv", index=False)
        best_j = sw.loc[sw.youden.idxmax()]
        # threshold that first reaches >=0.80 sensitivity (if any), and its FP cost
        hi = sw[sw.sens >= 0.80]
        s80 = hi.loc[hi.thr.idxmax()] if len(hi) else None
        print(f"\n══ head {h}  ({ev} → tasks.txt head)  n_pos={len(pos)} n_neg={len(neg)}  AUROC={auroc:.3f} ══")
        print(f"  pos score: median={np.median(pos):.3f} p90={np.percentile(pos,90):.3f} max={pos.max():.3f}")
        print(f"  neg score: median={np.median(neg):.3f} p90={np.percentile(neg,90):.3f} max={neg.max():.3f}")
        d05 = sw.iloc[(sw.thr - 0.5).abs().idxmin()]
        print(f"  @0.50 default : sens={d05.sens:.3f} spec={d05.spec:.3f} ppv={d05.ppv:.3f} (tp={int(d05.tp)} fp={int(d05.fp)})")
        print(f"  Youden-best   : thr={best_j.thr:.3f} sens={best_j.sens:.3f} spec={best_j.spec:.3f} "
              f"ppv={best_j.ppv:.3f} f1={best_j.f1:.3f} J={best_j.youden:.3f} (tp={int(best_j.tp)} fp={int(best_j.fp)})")
        if s80 is not None:
            print(f"  sens>=0.80 at : thr={s80.thr:.3f} sens={s80.sens:.3f} spec={s80.spec:.3f} ppv={s80.ppv:.3f} (fp={int(s80.fp)})")
        else:
            print(f"  sens>=0.80    : UNREACHABLE at any threshold (head does not rank positives high enough)")
        summary.append(dict(head=h, event=ev, n_pos=len(pos), n_neg=len(neg), auroc=round(auroc,3),
                            youden_thr=round(float(best_j.thr),3), youden_sens=round(float(best_j.sens),3),
                            youden_spec=round(float(best_j.spec),3), youden_ppv=round(float(best_j.ppv),3),
                            youden_f1=round(float(best_j.f1),3)))
    pd.DataFrame(summary).to_csv(f"{OUT}/recommended_thresholds.csv", index=False)
    print(f"\nArtifacts → {OUT}/")


if __name__ == "__main__":
    main()
