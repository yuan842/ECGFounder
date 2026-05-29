"""MOVE — synchronized non-overlapping 10-second segmentation (SEGMENTATION ONLY).

Implements data/MOVE/SEGMENTATION_PLAN.md. NO preprocessing: ECG windows are
saved RAW (no notch / bandpass / baseline / z-score). A separate downstream step
applies ECGPreprocessor before model input.

What this does:
  - Aligns the per-subject devices on a shared absolute-time clock (EDF starttime).
  - Builds a 10 s non-overlapping grid over the device intersection [T0, T1],
    plus one zero-padded tail window (paper §S3.3 policy).
  - Constructs each window by POSITION-AWARE zero-fill (real samples at their true
    time offset; gaps zero-filled in place → continuous, cross-sensor-aligned time axis).
  - Emits pad-aware, time-synchronized metadata: nominal vs real bounds, activity
    label by real-data midpoint, chest motion over real samples only, per-modality
    pad fractions.

v1 saved modalities (each tied to one device):
  ecg        ← chest  'ecg:gel'      500 Hz → (5000,)
  acc_chest  ← chest  'acc_chest:{x,y,z}' 500 Hz → (3, 5000)
  ppg_wrist  ← empatica 'ppg:wrist'  64 Hz  → (640,)   [omitted if empatica absent]

The grid is synchronized over the devices we actually draw modalities from
(chest + empatica). Forearm is not used in v1 so it does not constrain the grid
(this keeps e.g. H39D's 43-min-late forearm from shrinking the usable set).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from edf_reader import read_header, read_signal, read_annotations
from sqi import gravity_removed_motion   # fix #2: production gravity-removed motion def

WINDOW_S   = 10.0
_DC_REMOVAL_MIN = 5            # min real samples for the gravity-removal moving-average
ACTIVITY   = {"baseline", "lift", "greetings", "gesticulate",
              "walk_before", "run", "walk_after"}
MOVE_DIR   = Path("data/MOVE")
OUT_DIR    = Path("res/move_segmented")


# ─── per-device signal bundle ──────────────────────────────────────────────
def load_device(path: Path, channels: list[str]):
    """Return dict(label→(signal, fs)) + (start_abs, end_abs) for given channels."""
    hdr = read_header(str(path))
    start_abs = hdr.start_seconds
    end_abs   = start_abs + hdr.total_duration_s
    sigs = {}
    for ch in channels:
        if ch in hdr.labels:
            sigs[ch] = read_signal(str(path), hdr, ch)
    return sigs, start_abs, end_abs, hdr


def fill_window(signal: np.ndarray, fs: float, dev_start_abs: float,
                w_t0: float, win_len: int):
    """Position-aware zero-fill: place real samples at their true offset in a
    zero buffer of length win_len. Return (out, n_real) where n_real = #non-pad."""
    out = np.zeros(win_len, dtype=np.float32)
    dev_end_abs = dev_start_abs + len(signal) / fs
    w_t1 = w_t0 + WINDOW_S
    fill_t0 = max(w_t0, dev_start_abs)
    fill_t1 = min(w_t1, dev_end_abs)
    if fill_t1 <= fill_t0:
        return out, 0
    out_lo = int(round((fill_t0 - w_t0) * fs))
    src_lo = int(round((fill_t0 - dev_start_abs) * fs))
    n = int(round((fill_t1 - fill_t0) * fs))
    n = min(n, win_len - out_lo, len(signal) - src_lo)
    out[out_lo: out_lo + n] = signal[src_lo: src_lo + n]
    return out, n


def activity_at(t_abs: float, onsets_abs: list[tuple[float, str]]) -> str:
    """Label whose [onset, next_onset) interval contains t_abs."""
    label = "unknown"
    for onset, lab in onsets_abs:                       # onsets sorted ascending
        if onset <= t_abs:
            label = lab
        else:
            break
    return label


def segment_subject(subj: str, rows: list, ecg_lead: str):
    chest_p    = MOVE_DIR / subj / "scientisst_chest.edf"
    empatica_p = MOVE_DIR / subj / "empatica.edf"
    forearm_p  = MOVE_DIR / subj / "scientisst_forearm.edf"
    if not chest_p.exists():
        print(f"  [skip] {subj}: no chest device"); return

    # Load chest (ECG + ACC) — always the anchor
    chest_sigs, c_start, c_end, c_hdr = load_device(
        chest_p, [f"ecg:{ecg_lead}", "acc_chest:x", "acc_chest:y", "acc_chest:z"])
    used_starts, used_ends = [c_start], [c_end]
    devices_present = ["chest"]

    # Empatica (wrist PPG) — optional, constrains the grid when present
    emp_sigs = e_start = e_end = None
    if empatica_p.exists():
        emp_sigs, e_start, e_end, _ = load_device(empatica_p, ["ppg:wrist"])
        used_starts.append(e_start); used_ends.append(e_end)
        devices_present.append("empatica")
    if forearm_p.exists():
        devices_present.append("forearm")               # present but unused in v1

    # Intersection over USED devices (chest [+ empatica])
    T0 = max(used_starts)
    T1 = min(used_ends)
    overlap = T1 - T0
    if overlap < WINDOW_S:
        print(f"  [skip] {subj}: overlap {overlap:.1f}s < {WINDOW_S}s"); return

    # Activity onsets from the chest annotation channel → absolute time, sorted
    anns = read_annotations(str(chest_p), c_hdr)
    onsets_abs = sorted((c_start + o, l) for o, l in anns if l in ACTIVITY)

    n_full = int(overlap // WINDOW_S)
    remainder = overlap - n_full * WINDOW_S
    n_windows = n_full + (1 if remainder > 1e-6 else 0)

    ecg_label = f"ecg:{ecg_lead}"
    ecg_sig, ecg_fs = chest_sigs[ecg_label]
    accx, _ = chest_sigs["acc_chest:x"]
    accy, _ = chest_sigs["acc_chest:y"]
    accz, acc_fs = chest_sigs["acc_chest:z"]
    ppg_sig = ppg_fs = None
    if emp_sigs and "ppg:wrist" in emp_sigs:
        ppg_sig, ppg_fs = emp_sigs["ppg:wrist"]

    for d in ("ecg", "acc_chest", "ppg_wrist"):
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    for k in range(n_windows):
        w_t0 = T0 + k * WINDOW_S
        wid  = f"{subj}_w{k:04d}"

        # ECG (5000) — position-aware fill
        L_ecg = int(round(WINDOW_S * ecg_fs))
        ecg_win, ecg_real = fill_window(ecg_sig, ecg_fs, c_start, w_t0, L_ecg)
        ecg_pad = 1.0 - ecg_real / L_ecg

        # real-data span of the window (from the ECG channel) → drives label + bounds
        real_t0 = max(w_t0, c_start)
        real_t1 = min(w_t0 + WINDOW_S, c_end)
        real_mid = 0.5 * (real_t0 + real_t1)

        # chest ACC (3, 5000)
        L_acc = int(round(WINDOW_S * acc_fs))
        ax, ar = fill_window(accx, acc_fs, c_start, w_t0, L_acc)
        ay, _  = fill_window(accy, acc_fs, c_start, w_t0, L_acc)
        az, _  = fill_window(accz, acc_fs, c_start, w_t0, L_acc)
        acc_win = np.stack([ax, ay, az], axis=0)
        acc_pad = 1.0 - ar / L_acc
        # Motion over REAL samples only, using the PRODUCTION gravity-removed
        # dynamic-motion definition (fix #2-corrected: matches
        # multiclass_fp_suppression.extract_features, NOT raw mean magnitude).
        # → directly comparable to the deployed motion gates (5 / 15 / 24 mG).
        if ar >= _DC_REMOVAL_MIN:
            real_acc = np.stack([ax[:ar], ay[:ar], az[:ar]], axis=0)
            mot = gravity_removed_motion(real_acc)
            chest_motion_mg = round(mot["mean_motion_mg"], 3)     # = production mean_motion
            chest_motion_std_mg = round(mot["std_motion_mg"], 3)
        else:
            chest_motion_mg = chest_motion_std_mg = float("nan")

        # wrist PPG (640) — only if empatica present
        ppg_pad = float("nan"); ppg_path = ""
        if ppg_sig is not None:
            L_ppg = int(round(WINDOW_S * ppg_fs))
            ppg_win, ppg_real = fill_window(ppg_sig, ppg_fs, e_start, w_t0, L_ppg)
            ppg_pad = 1.0 - ppg_real / L_ppg
            ppg_path = f"ppg_wrist/{wid}.npy"
            np.save(OUT_DIR / ppg_path, ppg_win)

        # save raw windows (NO preprocessing)
        ecg_path = f"ecg/{wid}.npy"; acc_path = f"acc_chest/{wid}.npy"
        np.save(OUT_DIR / ecg_path, ecg_win)
        np.save(OUT_DIR / acc_path, acc_win)

        is_padded = (ecg_pad > 0) or (acc_pad > 0) or (ppg_sig is not None and ppg_pad > 0)
        rows.append(dict(
            window_id=wid, subject=subj, window_index=k,
            nominal_start_s=round(w_t0, 3), nominal_end_s=round(w_t0 + WINDOW_S, 3),
            real_start_s=round(real_t0, 3), real_end_s=round(real_t1, 3),
            activity_label=activity_at(real_mid, onsets_abs),
            sync_devices="+".join(["chest"] + (["empatica"] if emp_sigs else [])),
            devices_present="+".join(devices_present),
            ecg_lead=ecg_lead,
            ecg_path=ecg_path, acc_chest_path=acc_path, ppg_wrist_path=ppg_path,
            chest_motion_mg=None if (isinstance(chest_motion_mg, float) and np.isnan(chest_motion_mg)) else chest_motion_mg,
            chest_motion_std_mg=None if (isinstance(chest_motion_std_mg, float) and np.isnan(chest_motion_std_mg)) else chest_motion_std_mg,
            is_zero_padded=bool(is_padded),
            ecg_pad_fraction=round(ecg_pad, 4),
            acc_chest_pad_fraction=round(acc_pad, 4),
            ppg_wrist_pad_fraction=None if np.isnan(ppg_pad) else round(ppg_pad, 4),
        ))
    print(f"  {subj}: {n_windows} windows  (overlap {overlap:.0f}s, "
          f"devices={'+'.join(devices_present)}, tail_pad={'yes' if remainder>1e-6 else 'no'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["gel", "dry"], default="gel")
    ap.add_argument("--subjects", default="all",
                    help="comma-separated subject codes, or 'all'")
    args = ap.parse_args()

    subjects = (sorted(d.name for d in MOVE_DIR.iterdir()
                       if d.is_dir())
                if args.subjects == "all"
                else args.subjects.split(","))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list = []
    print(f"Segmenting MOVE (lead=ecg:{args.lead}, {WINDOW_S:.0f}s non-overlapping) ...")
    for subj in subjects:
        if (MOVE_DIR / subj).is_dir():
            segment_subject(subj, rows, args.lead)

    df = pd.DataFrame(rows)
    meta_path = OUT_DIR / "move_window_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"\nTotal windows: {len(df)}  across {df['subject'].nunique()} subjects")
    print(f"Metadata → {meta_path}")
    if len(df):
        print(f"Padded windows: {int(df['is_zero_padded'].sum())} "
              f"({100*df['is_zero_padded'].mean():.1f}%)")
        print("Activity distribution:")
        print(df['activity_label'].value_counts().to_string())


if __name__ == "__main__":
    main()
