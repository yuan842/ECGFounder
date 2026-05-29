"""SQI + motion characterization — TP (fzark) vs FP (doctor-removed) cohorts.

NO detection / model inference. Purely characterizes the SIGNAL: gravity-removed
motion + the full SQI panel (raw-stage + band-stage + composites) via sqi.compute_sqi,
and compares the distributions between the two cohorts, overall and per event.

Cohorts (stratified seed 42, ≤500/class — same alignment as the eval scripts):
  • TP = ecg_tp_fzark            (clinician-confirmed true events)
  • FP = ecg_fp_doctor removed1  (clinician-removed false positives)

Reader mirrors multiclass_fp_suppression.extract_features:
  ECG  → concat samples / magnification  (mV)
  ACC  → rows / 2048 (ACC_SCALE_FACTOR)  (g)   → sqi.gravity_removed_motion

Outputs (res/sqi_motion_tp_fp/):
  per_window.csv        — every window: cohort, event, all SQI+motion metrics
  summary_overall.csv   — TP vs FP median/IQR/mean/Cohen's d per metric
  summary_per_event.csv — same, split by event type
"""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np, pandas as pd
from tqdm import tqdm

from sqi import compute_sqi

TP_DIR = "data/ecg_tp_fzark"
FP_DIR = "data/ecg_fp_doctor removed1"
OUT = "res/sqi_motion_tp_fp"
ACC_SCALE = 2048.0          # fzark device raw→g
PER_CLASS = 500

# metrics to compare (SQI + motion only; mean_hr/rr_cv are signal descriptors, not detection)
MOTION = ["mean_motion_mg", "std_motion_mg", "max_motion_mg"]
RAW    = ["dynamic_range_mv", "flat_pct", "clip_pct", "baseline_drift", "hf_noise_ratio"]
BAND   = ["snr_proxy", "kurt", "rr_cv", "pct_physiologic_hr", "mean_hr_bpm"]
COMP   = ["sqi_score_ecg", "sqi_score_amb"]
METRICS = MOTION + RAW + BAND + COMP


def read_window(json_path):
    """Return (ecg_mv, fs, acc_g (3,T)|None) mirroring extract_features parsing."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    ecg, acc_rows, fs, mag = [], [], 128, 1000
    for item in data:
        if not isinstance(item, dict):
            continue
        d = item.get("data", {})
        a = d.get("acc")
        if a:
            for r in a:
                if isinstance(r, dict):
                    acc_rows.append([r.get("x", 0), r.get("y", 0), r.get("z", 0)])
                elif isinstance(r, (list, tuple)) and len(r) >= 3:
                    acc_rows.append([r[0], r[1], r[2]])
        e = d.get("ecg")
        if e is not None:
            fs = d.get("sf", fs) or fs
            mag = d.get("magnification", mag) or mag
            ecg.extend(e)
    if len(ecg) < fs * 3:
        return None
    ecg_mv = np.asarray(ecg, dtype=float) / float(mag)
    acc_g = (np.asarray(acc_rows, dtype=float) / ACC_SCALE).T if acc_rows else None
    return ecg_mv, int(fs), acc_g


def collect(summary_csv, data_dir, cohort):
    df = pd.read_csv(summary_csv)
    df = pd.concat([g.sample(n=min(PER_CLASS, len(g)), random_state=42)
                    for _, g in df.groupby("Event Type")]).reset_index(drop=True)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"  {cohort}", leave=False):
        jp = os.path.join(data_dir, str(r["JSON File"]).replace("\\", "/"))
        rd = read_window(jp)
        if rd is None:
            continue
        ecg_mv, fs, acc_g = rd
        try:
            panel = compute_sqi(ecg_mv, fs, acc_g)
        except Exception:
            continue
        row = {"cohort": cohort, "event": r["Event Type"]}
        for m in METRICS:
            row[m] = panel.get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def compare(df_tp, df_fp, label):
    rows = []
    for m in METRICS:
        tp = df_tp[m].to_numpy(float); fp = df_fp[m].to_numpy(float)
        tpf, fpf = tp[np.isfinite(tp)], fp[np.isfinite(fp)]
        rows.append(dict(
            group=label, metric=m,
            tp_n=len(tpf), fp_n=len(fpf),
            tp_median=np.median(tpf) if len(tpf) else np.nan,
            fp_median=np.median(fpf) if len(fpf) else np.nan,
            tp_iqr=f"[{np.percentile(tpf,25):.3g},{np.percentile(tpf,75):.3g}]" if len(tpf) else "",
            fp_iqr=f"[{np.percentile(fpf,25):.3g},{np.percentile(fpf,75):.3g}]" if len(fpf) else "",
            tp_mean=tpf.mean() if len(tpf) else np.nan,
            fp_mean=fpf.mean() if len(fpf) else np.nan,
            cohens_d=cohens_d(tp, fp),
        ))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Collecting SQI+motion panels (no detection) ...")
    tp = collect(f"{TP_DIR}/summary.csv", TP_DIR, "TP")
    fp = collect(f"{FP_DIR}/summary.csv", FP_DIR, "FP")
    allw = pd.concat([tp, fp], ignore_index=True)
    allw.to_csv(f"{OUT}/per_window.csv", index=False)
    print(f"  TP windows: {len(tp)}  FP windows: {len(fp)}")

    ov = compare(tp, fp, "OVERALL")
    ov.round(4).to_csv(f"{OUT}/summary_overall.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n" + "="*120)
    print("OVERALL — SQI + motion, TP vs FP (median, mean, Cohen's d = (TP−FP)/pooled_sd)")
    print("="*120)
    print(ov[["metric","tp_median","fp_median","tp_mean","fp_mean","cohens_d"]].to_string(index=False))

    # per-event (only events present in both cohorts)
    common = sorted(set(tp.event) & set(fp.event))
    per = []
    for ev in common:
        c = compare(tp[tp.event == ev], fp[fp.event == ev], ev)
        c["tp_windows"] = (tp.event == ev).sum(); c["fp_windows"] = (fp.event == ev).sum()
        per.append(c)
    per_df = pd.concat(per, ignore_index=True)
    per_df.round(4).to_csv(f"{OUT}/summary_per_event.csv", index=False)

    print("\n" + "="*120)
    print("PER-EVENT — motion (mean_motion_mg) and SQI composite (sqi_score_amb): TP vs FP")
    print("="*120)
    piv = per_df[per_df.metric.isin(["mean_motion_mg","snr_proxy","clip_pct","flat_pct",
                                     "sqi_score_ecg","sqi_score_amb"])]
    for ev in common:
        sub = piv[piv.group == ev]
        nt = int(sub.tp_n.iloc[0]); nf = int(sub.fp_n.iloc[0])
        print(f"\n── {ev}  (TP n={nt}, FP n={nf}) ──")
        print(sub[["metric","tp_median","fp_median","cohens_d"]].to_string(index=False))
    print(f"\nArtifacts → {OUT}/")


if __name__ == "__main__":
    main()
