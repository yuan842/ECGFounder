"""Re-evaluate base backbone vs DualHeadECGFounder on PTB-XL, FP-algo OFF/ON.

PTB-XL is fully labelled → supervised metrics (sens/spec/PPV/F1/AUROC) vs ground
truth, unlike the all-negative MOVE/fp_doctor cohorts.

Two facts that shape this eval:
  1. DualHead routes 8 PTB-XL-specific heads through the fuzzy checkpoint
     (idx 2,18,26,32,36,62,70,82). On ALL other heads it is identical to base.
     So the fine-tune effect appears ONLY on those 8 heads — where the fuzzy
     model was validated to improve (val_75deg ROC gains).
  2. The v2 FP-suppression algo is MOTION-based; PTB-XL has NO accelerometer.
     → AFib/SV-Trig/V-Trig motion gates are INAPPLICABLE. Only the Bradycardia
       HR-gate (HR ≤ 56.3, HR computable from ECG) can run. "FP on" is therefore
       near-vacuous on PTB-XL; reported for completeness.

Reproducible random sample of PTB-XL (seed 0). Lead II via for_ptbxl().
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np, pandas as pd, torch
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import roc_auc_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from preprocessing import ECGPreprocessor

PTBXL_ROOT = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
N_SAMPLE = 4000
FS = 500
FUZZY_HEADS = {2:"NORMAL ECG",18:"INCOMPLETE RBBB",26:"LVH",32:"ATRIAL FLUTTER",
               36:"LAFB",62:"INCOMPLETE LBBB",70:"LPFB",82:"RVH"}
BRADY_IDX = 4
RESULT = "res/ptbxl_eval2"


def hr_bpm(sig):
    try:
        b,a = butter(2,[5/(FS/2),20/(FS/2)],btype="bandpass"); q=filtfilt(b,a,sig)
        pk,_=find_peaks(q,distance=int(0.3*FS),height=np.std(q))
        if len(pk)<2: return np.nan
        return float(np.median(60.0/(np.diff(pk)/FS)))
    except Exception:
        return np.nan


def metrics(y, p, thr=0.5):
    fired = p>=thr
    tp=int((fired&(y==1)).sum()); fp=int((fired&(y==0)).sum())
    fn=int((~fired&(y==1)).sum()); tn=int((~fired&(y==0)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan
    spec=tn/(tn+fp) if tn+fp else np.nan
    ppv =tp/(tp+fp) if tp+fp else np.nan
    f1  =2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan
    auc =roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan
    return dict(n_pos=int((y==1).sum()),tp=tp,fp=fp,sens=sens,spec=spec,ppv=ppv,f1=f1,auroc=auc)


def main():
    os.makedirs(RESULT, exist_ok=True)
    lab = pd.read_csv("csv/ptbxl_label.csv").dropna(subset=["filename_hr","label"])
    samp = lab.sample(n=min(N_SAMPLE,len(lab)), random_state=0).reset_index(drop=True)
    Y = np.array([json.loads(x) for x in samp["label"]])           # (N,150) ground truth
    print(f"PTB-XL sample: {len(samp)} records (seed 0), {Y.shape[1]} labels")

    device = resolve_device()
    prep = ECGPreprocessor.for_ptbxl()
    print("Preprocessing + collecting HR ...")
    X, hrs = [], []
    for _, r in samp.iterrows():
        t = prep.from_wfdb(os.path.join(PTBXL_ROOT, r["filename_hr"]))   # (1,5000)
        X.append(t); hrs.append(hr_bpm(t.numpy().ravel()))
    X = torch.stack(X).to(device); hrs=np.array(hrs)

    def infer(m):
        m.eval(); out=[]
        with torch.no_grad():
            for i in range(0,len(X),128): out.append(torch.sigmoid(m(X[i:i+128])).cpu().numpy())
        return np.concatenate(out)
    print("Running BASE ..."); base=infer(load_ecgfounder(device))
    print("Running DualHead ..."); dual=infer(DualHeadECGFounder(device, routing="ptbxl_specific"))
    np.save(f"{RESULT}/base_probs.npy",base); np.save(f"{RESULT}/dual_probs.npy",dual); np.save(f"{RESULT}/gt.npy",Y)

    # ── 1. base vs DualHead on the 8 fuzzy-routed heads (supervised) ─────────
    print("\n"+"="*100)
    print("1. BASE vs DUALHEAD (fine-tune) on the 8 PTB-XL-specific heads — supervised, threshold 0.5")
    print("="*100)
    print(f"  {'head':<22}{'n_pos':>6} | {'BASE  auroc/sens/ppv/f1':>30} | {'DUAL  auroc/sens/ppv/f1':>30}")
    rows=[]
    for idx,nm in FUZZY_HEADS.items():
        mb=metrics(Y[:,idx],base[:,idx]); md=metrics(Y[:,idx],dual[:,idx])
        print(f"  {nm:<22}{mb['n_pos']:>6} | "
              f"{mb['auroc']:>6.3f}/{mb['sens']:.2f}/{mb['ppv']:.2f}/{mb['f1']:.2f}{'':>6} | "
              f"{md['auroc']:>6.3f}/{md['sens']:.2f}/{md['ppv']:.2f}/{md['f1']:.2f}")
        rows.append(dict(idx=idx,head=nm,n_pos=mb['n_pos'],
                         base_auroc=mb['auroc'],dual_auroc=md['auroc'],
                         base_f1=mb['f1'],dual_f1=md['f1'],
                         base_ppv=mb['ppv'],dual_ppv=md['ppv'],
                         d_auroc=md['auroc']-mb['auroc'], d_f1=md['f1']-mb['f1']))
    rdf=pd.DataFrame(rows); rdf.to_csv(f"{RESULT}/ptbxl_base_vs_dual_fuzzyheads.csv",index=False)
    print(f"\n  mean ΔAUROC (dual−base) on 8 heads: {rdf.d_auroc.mean():+.3f} | mean ΔF1: {rdf.d_f1.mean():+.3f}")

    # identity check on non-fuzzy heads
    nonf=[i for i in range(150) if i not in FUZZY_HEADS]
    print(f"  base vs dual max|Δ| on the other {len(nonf)} heads: {np.abs(base[:,nonf]-dual[:,nonf]).max():.2e} (expect 0 → identical)")

    # ── 2. FP-algo OFF/ON ────────────────────────────────────────────────────
    print("\n"+"="*100)
    print("2. FP-algo OFF vs ON on PTB-XL")
    print("="*100)
    print("  Motion gates (AFib≤5mG, SV-Trig≥15mG, V-Trig≥24mG) — INAPPLICABLE: PTB-XL has no accelerometer.")
    print("  Only applicable gate: Bradycardia (idx 4) suppress if HR > 56.3 bpm (HR from ECG).")
    yb=Y[:,BRADY_IDX]; pb=base[:,BRADY_IDX]; fired=pb>=0.5
    kept = fired & ~((hrs>56.3) & ~np.isnan(hrs))      # suppress fired alerts with HR>56.3
    m_off=metrics(yb,pb)
    tp_on=int((kept&(yb==1)).sum()); fp_on=int((kept&(yb==0)).sum())
    print(f"\n  Bradycardia (idx 4), n_pos={m_off['n_pos']}:")
    print(f"    FP algo OFF: TP={m_off['tp']}  FP={m_off['fp']}  sens={m_off['sens']:.3f}  ppv={m_off['ppv']:.3f}")
    print(f"    FP algo ON : TP={tp_on}  FP={fp_on}  sens={tp_on/max(1,m_off['n_pos']):.3f}  "
          f"ppv={tp_on/max(1,tp_on+fp_on):.3f}   (HR-gate)")
    print(f"\n  All other heads: FP algo ON == OFF (no applicable rule on PTB-XL).")
    print(f"\nArtifacts → {RESULT}/")


if __name__ == "__main__":
    main()
