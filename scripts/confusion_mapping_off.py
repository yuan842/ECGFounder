"""Baseline model, FP-algo OFF — comprehensive accuracy / FP analysis + confusion mapping.

Cohorts (same stratified alignment as scripts/combined_binary_eval.py:
random_state=42, per_class=500, grouped by 'Event Type'):
  • POSITIVES  = ecg_tp_fzark            (clinician-confirmed true events)
  • NEGATIVES  = ecg_fp_doctor removed1  (clinician-removed false positives)

Base ECGFounder probs are the cached OFF-suppression matrices (no re-inference):
  TP: res/tp_fzark_full_suppression/baseline_probs_tp.npy
  FP: res/fp_allclass_full_suppression/baseline_probs_full.npy

FP-algo OFF = raw sigmoid head outputs, no suppression. Threshold 0.5.

Outputs (res/confusion_off/):
  1. per_event_metrics_off.csv  — sens/spec/PPV/NPV/F1/accuracy/ROC-AUC/PR-AUC per event + overall
  2. tp_cofiring_pct.csv        — true event × fired fzark-head, % of windows (multi-label; rows can exceed 100%)
  3. fp_firing_pct.csv          — flagged(negative) event × fired fzark-head, % (residual FP firing)
  4. argmax_confusion.csv       — single-label argmax-over-fzark-heads confusion (square; rows sum to 100%)
  5. tp_cross_fire_full150.csv  — for each true event, top co-firing heads across the FULL 150-class space
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from label_config import FZARK_LABEL_MAP

OUT = "res/confusion_off"
TP_DIR = "data/ecg_tp_fzark"
FP_DIR = "data/ecg_fp_doctor removed1"
TP_PROBS = "res/tp_fzark_full_suppression/baseline_probs_tp.npy"
FP_PROBS = "res/fp_allclass_full_suppression/baseline_probs_full.npy"
THR = 0.5

# Unique fzark detector heads (10) — labelled by their 150-class name.
NAME = {int(r.founder_idx): r.founder_head for _, r in
        pd.read_csv("res/founder_ptbxl_fzark_mapping.csv").iterrows()}
def hname(i): return NAME.get(i, f"head_{i}")

UNIQUE_HEADS = sorted(set(FZARK_LABEL_MAP.values()))            # [4,5,6,9,16,19,68,93,98,142]
HEAD_COL = {h: f"{h}:{hname(h)}" for h in UNIQUE_HEADS}
# Event ordering for rows (stable, readable)
EVENTS = ['Atrial Fibrillation', 'Sinus Tachycardia', 'Bradycardia', 'ST Elevation',
          'Isolated Ventricular Beat', 'Ventricular Couplet', 'Ventricular Run',
          'Isolated Supraventricular Beat', 'Supraventricular Couplet',
          'Supraventricular Bigeminy', 'Supraventricular Trigeminy', 'Supraventricular Run',
          'Pause']


def load_cohort(summary_csv, probs_path, seed=42, per_class=500):
    df = pd.read_csv(summary_csv)
    sampled = [g.sample(n=min(per_class, len(g)), random_state=seed)
               for _, g in df.groupby('Event Type')]
    df = pd.concat(sampled).reset_index(drop=True)
    probs = np.load(probs_path)
    assert len(df) == probs.shape[0], (len(df), probs.shape)
    return df, probs


def per_event_metrics(tp_df, tp_p, fp_df, fp_p):
    rows = []
    pooled_pos, pooled_neg = [], []
    for et in EVENTS:
        if et not in FZARK_LABEL_MAP:
            continue
        h = FZARK_LABEL_MAP[et]
        ps = tp_p[(tp_df['Event Type'] == et).values, h]
        ns = fp_p[(fp_df['Event Type'] == et).values, h]
        n_pos, n_neg = len(ps), len(ns)
        tp = int((ps >= THR).sum()); fp = int((ns >= THR).sum())
        fn = n_pos - tp; tn = n_neg - fp
        if n_pos and n_neg:
            pooled_pos.append(ps); pooled_neg.append(ns)
        rows.append(dict(
            event=et, head=h, n_pos=n_pos, n_neg=n_neg, TP=tp, FP=fp, FN=fn, TN=tn,
            sens=tp/max(1, n_pos), spec=tn/max(1, n_neg),
            ppv=tp/max(1, tp+fp), npv=tn/max(1, tn+fn),
            accuracy=(tp+tn)/max(1, n_pos+n_neg),
            f1=(2*tp)/max(1, 2*tp+fp+fn),
            roc_auc=roc_auc_score(np.r_[np.ones(n_pos), np.zeros(n_neg)], np.r_[ps, ns])
                    if n_pos and n_neg else np.nan,
            pr_auc=average_precision_score(np.r_[np.ones(n_pos), np.zeros(n_neg)], np.r_[ps, ns])
                    if n_pos and n_neg else np.nan,
        ))
    # micro-pooled overall
    ps = np.concatenate(pooled_pos); ns = np.concatenate(pooled_neg)
    n_pos, n_neg = len(ps), len(ns)
    tp = int((ps >= THR).sum()); fp = int((ns >= THR).sum()); fn = n_pos-tp; tn = n_neg-fp
    rows.append(dict(
        event='OVERALL (micro)', head=-1, n_pos=n_pos, n_neg=n_neg, TP=tp, FP=fp, FN=fn, TN=tn,
        sens=tp/n_pos, spec=tn/n_neg, ppv=tp/max(1, tp+fp), npv=tn/max(1, tn+fn),
        accuracy=(tp+tn)/(n_pos+n_neg), f1=(2*tp)/max(1, 2*tp+fp+fn),
        roc_auc=roc_auc_score(np.r_[np.ones(n_pos), np.zeros(n_neg)], np.r_[ps, ns]),
        pr_auc=average_precision_score(np.r_[np.ones(n_pos), np.zeros(n_neg)], np.r_[ps, ns])))
    return pd.DataFrame(rows)


def cofiring_matrix(df, probs):
    """rows=event, cols=fzark heads + (none) ; value = % of event's windows firing that head."""
    fired = probs >= THR
    rows = []
    for et in EVENTS:
        m = (df['Event Type'] == et).values
        n = int(m.sum())
        if n == 0:
            continue
        row = {'event': et, 'n': n}
        none_mask = np.ones(n, dtype=bool)
        for h in UNIQUE_HEADS:
            f = fired[m, h]
            row[HEAD_COL[h]] = 100.0 * f.mean()
            none_mask &= ~f
        row['(none fired)'] = 100.0 * none_mask.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index('event')


def argmax_confusion(df, probs, true_is_negative=False):
    """Single-label: predicted = argmax over the 10 fzark heads if its prob>=THR else NONE.
    rows = true event (or NEGATIVE), cols = predicted head-name or NONE. Counts."""
    head_arr = np.array(UNIQUE_HEADS)
    sub = probs[:, head_arr]                       # (N,10)
    amax = sub.argmax(axis=1)
    amax_prob = sub.max(axis=1)
    pred = np.where(amax_prob >= THR,
                    [HEAD_COL[head_arr[a]] for a in amax], 'NONE')
    cols = [HEAD_COL[h] for h in UNIQUE_HEADS] + ['NONE']
    rows = []
    iter_rows = ['NEGATIVE'] if true_is_negative else EVENTS
    for et in iter_rows:
        m = np.ones(len(df), bool) if true_is_negative else (df['Event Type'] == et).values
        n = int(m.sum())
        if n == 0:
            continue
        r = {'true': et, 'n': n}
        p = pred[m]
        for c in cols:
            r[c] = int((p == c).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index('true')


def cross_fire_full150(df, probs, topk=6):
    """For each true event, the top co-firing heads across ALL 150 classes (% of windows)."""
    fired = probs >= THR
    out = []
    for et in EVENTS:
        m = (df['Event Type'] == et).values
        n = int(m.sum())
        if n == 0:
            continue
        rate = fired[m].mean(axis=0) * 100.0
        own = FZARK_LABEL_MAP[et]
        order = np.argsort(-rate)
        top = [(int(i), hname(int(i)), round(float(rate[i]), 1))
               for i in order if rate[i] > 0][:topk]
        out.append(dict(event=et, n=n, own_head=f"{own}:{hname(own)}",
                        own_fire_pct=round(float(rate[own]), 1),
                        top_cofiring=" | ".join(f"{nm}({pc}%)" for _, nm, pc in top)))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    tp_df, tp_p = load_cohort(f"{TP_DIR}/summary.csv", TP_PROBS)
    fp_df, fp_p = load_cohort(f"{FP_DIR}/summary.csv", FP_PROBS)
    print(f"TP windows: {len(tp_df)}  FP windows: {len(fp_df)}  (base model, FP-algo OFF, thr={THR})\n")

    met = per_event_metrics(tp_df, tp_p, fp_df, fp_p)
    met.round(3).to_csv(f"{OUT}/per_event_metrics_off.csv", index=False)
    pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f"{x:.3f}")
    print("="*120); print("PER-EVENT ACCURACY / FP METRICS  (base, FP-OFF, t=0.5)"); print("="*120)
    print(met[['event','n_pos','n_neg','TP','FP','FN','TN','sens','spec','ppv','npv',
               'accuracy','f1','roc_auc','pr_auc']].to_string(index=False))

    tp_co = cofiring_matrix(tp_df, tp_p); tp_co.round(1).to_csv(f"{OUT}/tp_cofiring_pct.csv")
    fp_co = cofiring_matrix(fp_df, fp_p); fp_co.round(1).to_csv(f"{OUT}/fp_firing_pct.csv")
    print("\n"+"="*120); print("TP CO-FIRING (% of each TRUE event's windows firing each fzark head; multi-label → rows can exceed 100%)"); print("="*120)
    print(tp_co.round(1).to_string())
    print("\n"+"="*120); print("FP RESIDUAL FIRING (% of each CLINICIAN-REMOVED event's windows still firing each fzark head)"); print("="*120)
    print(fp_co.round(1).to_string())

    tp_cm = argmax_confusion(tp_df, tp_p); tp_cm.to_csv(f"{OUT}/argmax_confusion_tp.csv")
    fp_cm = argmax_confusion(fp_df, fp_p, true_is_negative=True); fp_cm.to_csv(f"{OUT}/argmax_confusion_fp.csv")
    print("\n"+"="*120); print("ARGMAX SINGLE-LABEL CONFUSION — TP cohort (pred = argmax over 10 fzark heads if >=0.5 else NONE)"); print("="*120)
    print(tp_cm.to_string())

    cf = cross_fire_full150(tp_df, tp_p); cf.to_csv(f"{OUT}/tp_cross_fire_full150.csv", index=False)
    print("\n"+"="*120); print("TOP CO-FIRING HEADS ACROSS FULL 150-CLASS SPACE (per true event)"); print("="*120)
    print(cf.to_string(index=False))
    print(f"\nArtifacts → {OUT}/")


if __name__ == "__main__":
    main()
