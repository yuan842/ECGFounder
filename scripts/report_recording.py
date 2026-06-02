"""Per-recording diagnosis report — assemble per-window scans into one summary.

Turns the segment→scan pipeline into a clinical-style report for a whole recording:
  1. run ECGFounder on each segmented window (raw → ECGPreprocessor → model)
  2. restrict to the 7-label DETECTION_SCOPE at per-head thresholds (head_threshold)
  3. apply FP suppression (device profile: motion/SQI gates) per window
  4. join window timestamps (real_start_s/real_end_s)
  5. emit ONE summary per recording:
       • per-window table (10 s resolution)
       • merged event intervals (onset/offset, gap-tolerant, min-duration)
       • recording-level burden rollup (per label: % time, episodes, longest)

Worked example: a MOVE subject (chest ecg:gel). MOVE is rhythm-NEGATIVE (activity only),
so any detections are effectively false positives — which makes this a clean demo of both
the report format AND the FP gate's effect (raw-fired vs surviving suppression).

Usage:
  python scripts/report_recording.py --subject 3B8D [--device move_chest_gel] [--model base]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, torch

from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from device_utils import resolve_device
from sqi import compute_sqi
import label_config as L
from fp_suppression import FPSuppressionPipeline
from overlay.signal_quality_gate import SignalQualityGate

SEG = "res/move_segmented"
OUT = "res/recording_reports"
FS = 500
HEAD2EVENT = {h: e for e, h in L.SCOPE_EVENT_TO_HEAD.items()}   # {2:'Normal ECG',5:'Atrial Fibrillation',...}
SCOPE = sorted(L.scope_indices())


def merge_events(t0, t1, fired, probs, max_gap=1, min_win=1):
    """Merge consecutive fired windows into episodes (gap-tolerant, min-duration)."""
    f = fired.copy()
    idx = np.where(f)[0]
    if len(idx) == 0:
        return []
    for a, b in zip(idx[:-1], idx[1:]):           # fill short dropouts
        if 0 < b - a - 1 <= max_gap:
            f[a+1:b] = True
    ev, i, n = [], 0, len(f)
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) >= min_win:
                ev.append(dict(onset_s=round(float(t0[i]), 1), offset_s=round(float(t1[j-1]), 1),
                               duration_s=round(float(t1[j-1] - t0[i]), 1),
                               n_windows=int(j - i), n_fired=int(fired[i:j].sum()),
                               peak_prob=round(float(probs[i:j].max()), 3),
                               mean_prob=round(float(probs[i:j].mean()), 3)))
            i = j
        else:
            i += 1
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="3B8D")
    ap.add_argument("--device", default="move_chest_gel", help="FP-suppression device profile")
    ap.add_argument("--model", default="base", choices=["base"])
    ap.add_argument("--max-gap", type=int, default=1, help="windows of dropout tolerated within an episode")
    ap.add_argument("--min-win", type=int, default=1, help="min windows for an episode")
    a = ap.parse_args()
    outdir = os.path.join(OUT, a.subject); os.makedirs(outdir, exist_ok=True)

    meta = pd.read_csv(f"{SEG}/move_window_metadata.csv")
    sub = meta[meta.subject == a.subject].sort_values("window_index").reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"no windows for subject {a.subject}; have {sorted(meta.subject.unique())}")
    print(f"subject {a.subject}: {len(sub)} windows  ({sub.real_start_s.min():.0f}-{sub.real_end_s.max():.0f}s)")

    # preprocess + per-window features
    prep = ECGPreprocessor(powerline_hz=50, normalize="winsorize")
    X, hr, snr, hfn, drift = [], [], [], [], []
    for _, r in sub.iterrows():
        raw = np.load(os.path.join(SEG, r["ecg_path"])).astype(np.float64)
        X.append(prep.process(raw, fs_in=FS))
        panel = compute_sqi(raw, FS, acc_xyz=None)
        hr.append(panel.get("mean_hr_bpm", np.nan)); snr.append(panel.get("snr_proxy", np.nan))
        hfn.append(panel.get("hf_noise_ratio", np.nan)); drift.append(panel.get("baseline_drift", np.nan))
    X = torch.stack(X).to(resolve_device())

    model = load_ecgfounder(resolve_device()); model.eval()
    probs = np.zeros((len(sub), 150), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 128):
            probs[s:s+128] = torch.sigmoid(model(X[s:s+128])).cpu().numpy()

    # S0 signal-quality gate (global policy, default ON) + motion FP suppression.
    # Noisy windows (low SNR or high baseline drift) are classified Noisy and skip
    # detection entirely; the FP-suppression SQI family is now off (motion only).
    sqg = SignalQualityGate()
    pipe = FPSuppressionPipeline(a.device)                    # motion-only (sqi default OFF)
    t0 = sub.real_start_s.to_numpy(float); t1 = sub.real_end_s.to_numpy(float)
    motion = sub.chest_motion_mg.to_numpy(float)
    fired = {h: np.zeros(len(sub), bool) for h in SCOPE}      # after gate + suppression
    raw_fired = {h: np.zeros(len(sub), bool) for h in SCOPE}  # before gate + suppression
    noisy_gated = np.zeros(len(sub), bool)
    rows = []
    for i in range(len(sub)):
        feats = {"mean_motion": motion[i], "mean_hr_bpm": hr[i], "snr_proxy": snr[i],
                 "hf_noise": hfn[i]}
        row = {"window_id": sub.window_id[i], "t_start_s": round(t0[i], 1), "t_end_s": round(t1[i], 1),
               "activity": sub.activity_label[i], "motion_mg": round(float(motion[i]), 2)}
        # S0 SQI gate — classify Noisy and skip detection
        g = sqg.gate(None, sqi={"snr_proxy": snr[i], "baseline_drift": drift[i]})
        noisy = not g.passed
        noisy_gated[i] = noisy
        det = []
        for h in SCOPE:
            p = float(probs[i, h]); row[f"p_{h}"] = round(p, 3)
            if noisy:
                continue                                      # skip detection on Noisy
            rf = p >= L.head_threshold(h); raw_fired[h][i] = rf
            keep = pipe.suppress(HEAD2EVENT[h], feats).keep if rf else False
            fired[h][i] = rf and keep
            if rf and not keep:
                det.append(f"{HEAD2EVENT[h]}(suppressed)")
            elif fired[h][i]:
                det.append(HEAD2EVENT[h])
        row["noisy_gated"] = bool(noisy)
        row["detections"] = "Noisy (gated)" if noisy else ("; ".join(det) if det else "-")
        rows.append(row)
    print(f"S0 SQG: {int(noisy_gated.sum())}/{len(sub)} windows classified Noisy → detection skipped")
    win_df = pd.DataFrame(rows)
    win_df.to_csv(f"{outdir}/windows.csv", index=False)

    # merged events + rollup
    rec_dur = float(t1.max() - t0.min())
    events, summary = [], []
    for h in SCOPE:
        ev = merge_events(t0, t1, fired[h], probs[:, h], a.max_gap, a.min_win)
        for e in ev:
            e["label"] = HEAD2EVENT[h]; e["head"] = h
        events += ev
        burden = sum(e["duration_s"] for e in ev)
        raw_n = int(raw_fired[h].sum()); kept_n = int(fired[h].sum())
        summary.append(dict(label=HEAD2EVENT[h], head=h, episodes=len(ev),
                            burden_s=round(burden, 1), burden_pct=round(100*burden/rec_dur, 1),
                            longest_s=round(max((e["duration_s"] for e in ev), default=0.0), 1),
                            windows_fired_raw=raw_n, windows_fired_after_fp=kept_n))
    events.sort(key=lambda e: e["onset_s"])
    summ_df = pd.DataFrame(summary)
    summ_df.to_csv(f"{outdir}/summary.csv", index=False)
    pd.DataFrame(events).to_csv(f"{outdir}/events.csv", index=False)

    report = dict(recording=dict(subject=a.subject, n_windows=len(sub), duration_s=round(rec_dur, 1),
                                 window_s=10, device=a.device, model=a.model,
                                 scope=sorted(L.scope_events())),
                  summary=summary, events=events)
    json.dump(report, open(f"{outdir}/report.json", "w"), indent=2)

    # human-readable
    md = [f"# Recording report — {a.subject}", "",
          f"- Duration **{rec_dur:.0f} s** ({rec_dur/60:.1f} min), {len(sub)} × 10 s windows",
          f"- Model **{a.model}** · FP device profile **{a.device}** · scope {SCOPE}",
          f"- ⚠ MOVE is rhythm-NEGATIVE (activity only) — detections here are effectively false positives.",
          "", "## Per-label burden (after FP suppression)", "",
          "| label | episodes | burden | % rec | longest | windows raw→FP |", "|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['label']} | {s['episodes']} | {s['burden_s']:.0f}s | {s['burden_pct']}% | "
                  f"{s['longest_s']:.0f}s | {s['windows_fired_raw']}→{s['windows_fired_after_fp']} |")
    md += ["", "## Event intervals (after FP suppression)", ""]
    if events:
        md.append("| onset | offset | dur | label | peak p |"); md.append("|---|---|---|---|---|")
        for e in events:
            md.append(f"| {e['onset_s']:.0f}s | {e['offset_s']:.0f}s | {e['duration_s']:.0f}s | {e['label']} | {e['peak_prob']} |")
    else:
        md.append("_No events survived suppression._")
    open(f"{outdir}/report.md", "w").write("\n".join(md))

    print("\n".join(md[:6]))
    print("\nper-label (raw→afterFP windows):")
    print(summ_df[["label","windows_fired_raw","windows_fired_after_fp","episodes","burden_pct"]].to_string(index=False))
    print(f"\nArtifacts → {outdir}/ (report.json, windows.csv, events.csv, summary.csv, report.md)")


if __name__ == "__main__":
    main()
