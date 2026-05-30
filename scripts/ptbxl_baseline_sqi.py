"""PTB-XL — baseline model (FP OFF) performance on the 7 scope labels +
comprehensive ECG SQI analysis. Full dataset (21,799 records).

Two facts:
  • PTB-XL has NO accelerometer → the FP-suppression algo is inapplicable.
    "FP off" = the raw base model (the only meaningful state here). SQI is
    ECG-ONLY (no motion / sqi_score_amb).
  • Scope = 7 labels {2,4,5,6,93,98,142}, all at threshold 0.5 (head_threshold).

Each record is read ONCE (raw lead II, mV, 500 Hz) and used for both the SQI panel
(sqi.compute_sqi, acc=None) and the model input (ECGPreprocessor.process).

Outputs (res/ptbxl_baseline_sqi/):
  base_probs_full.npy           (gitignored)  — (N,150) base sigmoid probs
  ptbxl_sqi_full.csv            — per-record ECG SQI panel + GT scope flags
  perf_scope.csv                — per-scope-label FP-off metrics
  sqi_overall.csv / sqi_by_label.csv
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

import wfdb
from sqi import compute_sqi
from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from device_utils import resolve_device
from label_config import SCOPE_EVENT_TO_HEAD, head_threshold

ROOT = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT = "res/ptbxl_baseline_sqi"
FS = 500
LEAD_II = 1
SQI_KEYS = ["dynamic_range_mv", "flat_pct", "clip_pct", "baseline_drift", "hf_noise_ratio",
            "snr_proxy", "kurt", "mean_hr_bpm", "rr_cv", "pct_physiologic_hr", "sqi_score_ecg"]
# scope labels in a readable order
SCOPE = [("Normal ECG", 2), ("Atrial Fibrillation", 5), ("Bradycardia", 4),
         ("Sinus Tachycardia", 6), ("Supraventricular Run", 93),
         ("Ventricular Run", 98), ("Pause", 142)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    lab = pd.read_csv("csv/ptbxl_label.csv").dropna(subset=["filename_hr", "label"]).reset_index(drop=True)
    if args.limit:
        lab = lab.iloc[:args.limit].reset_index(drop=True)
    N = len(lab)
    Y = np.array([json.loads(x) for x in lab["label"]])          # (N,150) GT
    print(f"PTB-XL: {N} records | lead II @ {FS} Hz | FP-off (no accelerometer) | ECG-only SQI", flush=True)

    prep = ECGPreprocessor.for_ptbxl()
    sqi_rows, X = [], np.zeros((N, 5000), dtype=np.float32)
    for i, fn in enumerate(lab["filename_hr"]):
        try:
            sig = wfdb.rdrecord(os.path.join(ROOT, fn), channels=[LEAD_II]).p_signal[:, 0]
            panel = compute_sqi(sig, FS, acc_xyz=None)
            t = prep.process(sig, FS, source_leads=["II"]).numpy().ravel()
            X[i, :len(t)] = t[:5000]
            sqi_rows.append({k: panel.get(k, np.nan) for k in SQI_KEYS})
        except Exception:
            sqi_rows.append({k: np.nan for k in SQI_KEYS})
        if (i + 1) % 2000 == 0:
            print(f"  read+SQI {i+1}/{N}", flush=True)

    sqi = pd.DataFrame(sqi_rows)
    for name, h in SCOPE:
        sqi[f"gt_{h}"] = Y[:, h]
    sqi.to_csv(f"{OUT}/ptbxl_sqi_full.csv", index=False)

    # ── model inference (FP off = raw base) ──
    device = resolve_device(); model = load_ecgfounder(device); model.eval()
    xb = torch.from_numpy(X).unsqueeze(1).to(device)
    probs = np.zeros((N, 150), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, 256):
            probs[s:s+256] = torch.sigmoid(model(xb[s:s+256])).cpu().numpy()
            if (s + 256) % 4096 == 0:
                print(f"  infer {min(s+256,N)}/{N}", flush=True)
    np.save(f"{OUT}/base_probs_full.npy", probs)

    # ── performance per scope label (FP off, per-head threshold) ──
    rows = []
    for name, h in SCOPE:
        y = Y[:, h]; p = probs[:, h]; thr = head_threshold(h)
        fired = p >= thr
        tp = int((fired & (y == 1)).sum()); fp = int((fired & (y == 0)).sum())
        fn = int((~fired & (y == 1)).sum()); tn = int((~fired & (y == 0)).sum())
        npos = int((y == 1).sum())
        rows.append(dict(label=name, head=h, thr=thr, n_pos=npos, n_neg=N-npos,
            TP=tp, FP=fp, FN=fn, TN=tn,
            sens=tp/max(1, tp+fn), spec=tn/max(1, tn+fp), ppv=tp/max(1, tp+fp),
            f1=2*tp/max(1, 2*tp+fp+fn),
            auroc=roc_auc_score(y, p) if npos and npos < N else np.nan,
            pr_auc=average_precision_score(y, p) if npos and npos < N else np.nan))
    perf = pd.DataFrame(rows); perf.round(4).to_csv(f"{OUT}/perf_scope.csv", index=False)

    # ── SQI summaries ──
    def summ(df):
        return {k: (float(np.nanmedian(df[k])), float(np.nanpercentile(df[k], 25)),
                    float(np.nanpercentile(df[k], 75))) for k in SQI_KEYS}
    ov = pd.DataFrame({k: [np.nanmedian(sqi[k]), np.nanpercentile(sqi[k], 25),
                           np.nanpercentile(sqi[k], 75), np.nanmean(sqi[k])] for k in SQI_KEYS},
                      index=["median", "p25", "p75", "mean"]).T
    ov.round(4).to_csv(f"{OUT}/sqi_overall.csv")
    by = {}
    for name, h in SCOPE:
        sub = sqi[sqi[f"gt_{h}"] == 1]
        if len(sub) >= 5:
            by[name] = {k: round(float(np.nanmedian(sub[k])), 3) for k in SQI_KEYS}
            by[name]["n"] = len(sub)
    by_df = pd.DataFrame(by).T
    by_df.to_csv(f"{OUT}/sqi_by_label.csv")

    pd.set_option("display.width", 230); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n" + "="*110)
    print("BASELINE PERFORMANCE (FP off — no accelerometer on PTB-XL), 7 scope labels, t=0.5")
    print("="*110)
    print(perf[["label","head","n_pos","sens","spec","ppv","f1","auroc","pr_auc"]].to_string(index=False))
    print("\n" + "="*110); print("ECG SQI — OVERALL (median / IQR / mean)"); print("="*110)
    print(ov.round(3).to_string())
    print("\n" + "="*110); print("ECG SQI — median by scope label"); print("="*110)
    print(by_df.to_string())
    print(f"\nArtifacts → {OUT}/")


if __name__ == "__main__":
    main()
