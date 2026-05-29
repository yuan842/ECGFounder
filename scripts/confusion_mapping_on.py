"""Baseline model, FP-algo ON — accuracy / FP analysis + confusion mapping.

Counterpart to confusion_mapping_off.py. Same cohorts, scope (DETECTION_SCOPE
{4,5,6,93,98,142}) and per-head thresholds — but with FP suppression APPLIED.

Within the scope, the only in-scope ACTIVE suppression rules are:
  • Atrial Fibrillation (head 5): motion gate  (mean_motion <= 5 mG → keep)
  • Bradycardia        (head 4): HR gate      (mean_hr_bpm <= 56.3 → keep)
SV-Trig/V-Trig rules are out of scope (suppressor passes them through), and heads
6/93/98/142 have no rule — so ON differs from OFF only in columns 4 and 5.

Builds a SUPPRESSED probability matrix (base probs with the AFib/Brady head scores
zeroed where the gate fails — read per-window from JSON for motion/HR), then reuses
the confusion functions from confusion_mapping_off.

Outputs → res/confusion_on/.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, pandas as pd
from tqdm import tqdm

import confusion_mapping_off as off          # reuse scope constants + confusion fns
from label_config import FZARK_LABEL_MAP, head_threshold
from multiclass_fp_suppression import MultiClassFPSuppressor

OUT = "res/confusion_on"
# in-scope events that have an active suppression rule (head ∈ DETECTION_SCOPE)
SUPP_EVENTS = {'Atrial Fibrillation': 5, 'Bradycardia': 4}


def suppress_matrix(df, base, data_dir, supp):
    """Return a copy of `base` with AFib(5)/Brady(4) head scores zeroed where the
    gate fails. Only firing windows (score >= head threshold) are processed."""
    out = base.copy()
    for event, head in SUPP_EVENTS.items():
        thr = head_threshold(head)
        idx = np.where(base[:, head] >= thr)[0]          # only firing windows matter
        for i in tqdm(idx, desc=f"  {event[:18]:<18} head{head}", leave=False):
            jp = os.path.join(data_dir, str(df.iloc[i]['JSON File']).replace('\\', '/'))
            try:
                r = supp.suppress_alert(event, float(base[i, head]), jp)
                if not r.keep:
                    out[i, head] = 0.0
            except Exception:
                pass
        n_supp = int((base[idx, head] >= thr).sum() - (out[idx, head] >= thr).sum())
        print(f"  {event:<22} head {head}: {len(idx)} firing → {n_supp} suppressed")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tp_df, tp_b = off.load_cohort(f"{off.TP_DIR}/summary.csv", off.TP_PROBS)
    fp_df, fp_b = off.load_cohort(f"{off.FP_DIR}/summary.csv", off.FP_PROBS)
    print(f"TP windows: {len(tp_df)}  FP windows: {len(fp_df)}  (base, FP-algo ON, scope "
          f"{sorted(off.scope_indices())}, per-head thresholds)\n")

    supp = MultiClassFPSuppressor(enabled=True)
    print("Applying suppression to TP cohort ..."); tp_p = suppress_matrix(tp_df, tp_b, off.TP_DIR, supp)
    print("Applying suppression to FP cohort ..."); fp_p = suppress_matrix(fp_df, fp_b, off.FP_DIR, supp)

    pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f"{x:.3f}")
    met = off.per_event_metrics(tp_df, tp_p, fp_df, fp_p)
    met.round(3).to_csv(f"{OUT}/per_event_metrics_on.csv", index=False)
    print("\n"+"="*120); print("PER-EVENT ACCURACY / FP METRICS  (base, FP-ON, scope, per-head thresholds)"); print("="*120)
    print(met[['event','n_pos','n_neg','TP','FP','sens','spec','ppv','npv','accuracy','f1','roc_auc','pr_auc']].to_string(index=False))

    tp_co = off.cofiring_matrix(tp_df, tp_p); tp_co.round(1).to_csv(f"{OUT}/tp_cofiring_pct.csv")
    fp_co = off.cofiring_matrix(fp_df, fp_p); fp_co.round(1).to_csv(f"{OUT}/fp_firing_pct.csv")
    print("\n"+"="*120); print("TP CO-FIRING (% of each TRUE event's windows firing each scope head; FP-ON)"); print("="*120)
    print(tp_co.round(1).to_string())
    print("\n"+"="*120); print("FP RESIDUAL FIRING (% of clinician-removed windows still firing each scope head; FP-ON)"); print("="*120)
    print(fp_co.round(1).to_string())

    tp_cm = off.argmax_confusion(tp_df, tp_p); tp_cm.to_csv(f"{OUT}/argmax_confusion_tp.csv")
    fp_cm = off.argmax_confusion(fp_df, fp_p, true_is_negative=True); fp_cm.to_csv(f"{OUT}/argmax_confusion_fp.csv")
    print("\n"+"="*120); print("ARGMAX SINGLE-LABEL CONFUSION — TP cohort (FP-ON)"); print("="*120)
    print(tp_cm.to_string())
    print("\n"+"="*120); print("ARGMAX on NEGATIVE pool (FP-ON)"); print("="*120)
    print(fp_cm.to_string())

    cf = off.cross_fire_full150(tp_df, tp_p); cf.to_csv(f"{OUT}/tp_cross_fire_full150.csv", index=False)
    print(f"\nArtifacts → {OUT}/")


if __name__ == "__main__":
    main()
