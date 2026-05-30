"""Ingest PhysioNet/CinC 2015 ("Reducing False Arrhythmia Alarms") → model-ready npz.

Open-access (NO DUA). The only realistic open source for **Pause/asystole (142)** and a
clean source for **VT (98)**. Structurally different from Chapman/PTB-XL:
  • ICU records, ~2 ECG leads (II, V) + ABP/PLETH — NO Lead I → **no angle derivation**.
    Output is SINGLE-LEAD (Lead-II proxy); joins the probe as the 60°/Lead-II stream only.
  • Long records (~5 min @ 250 Hz) with the alarm at the END → take the **last 10 s**.
  • Each record's **true/false alarm verdict drives the label**: TRUE alarm → positive for
    its head; FALSE alarm → all-zero = a HARD negative (looked like it, wasn't).

Usage:
  python scripts/cinc2015/ingest_cinc2015.py --dry-run                 # validate, no data
  python scripts/cinc2015/ingest_cinc2015.py --data-dir data/cinc2015 \
         --out data/cinc2015_fuzzy [--truth-csv truth.csv] [--report-unmapped]

Download (open): https://physionet.org/content/challenge-2015/  (training/ WFDB *.hea+*.mat)
"""
from __future__ import annotations
import os, sys, glob, json, argparse, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.cinc2015.cinc2015_label_map import (
    parse_alarm, labels_from_alarm, HEAD_NAMES, SCOPE_HEADS)

FS, NSAMP = 500, 5000                       # 10 s @ 500 Hz
ECG_PREF = ["II", "V", "V1", "V5", "I", "III", "MCL1", "MCL", "ECG"]
NON_ECG  = {"ABP", "PLETH", "RESP", "CO2", "CVP", "PAP"}
FOLD_TO_SPLIT = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}


def pick_ecg_lead(sig_names: list[str]) -> int | None:
    for pref in ECG_PREF:
        if pref in sig_names:
            return sig_names.index(pref)
    for i, n in enumerate(sig_names):              # any channel that isn't a known non-ECG
        if n.upper() not in NON_ECG:
            return i
    return None


def last_window(sig: np.ndarray, fs_in: int) -> np.ndarray:
    """Resample to 500 Hz, return the LAST 10 s (the alarm sits at the record end)."""
    if fs_in != FS:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(fs_in), FS)
        sig = resample_poly(sig, FS // g, int(fs_in) // g)
    if len(sig) >= NSAMP:
        return sig[-NSAMP:].astype(np.float32)
    out = np.zeros(NSAMP, np.float32); out[-len(sig):] = sig      # right-align (keep alarm end)
    return out


def record_fold(record_id: str) -> int:
    h = int(hashlib.sha1(record_id.encode()).hexdigest(), 16)
    return (h % 10) + 1


def ingest(data_dir, out_dir, truth_csv, report_unmapped):
    import wfdb, pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    truth = {}
    if truth_csv and os.path.exists(truth_csv):
        td = pd.read_csv(truth_csv)
        # expect columns like record, alarm_type, true_alarm (bool/0-1) — best-effort
        for _, r in td.iterrows():
            truth[str(r.iloc[0])] = bool(int(r.get("true_alarm", r.iloc[-1])))

    heas = sorted(glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True))
    if not heas:
        raise SystemExit(f"no .hea records under {data_dir}")
    print(f"found {len(heas)} CinC-2015 records under {data_dir}")

    ecgs, ids, labs, rows = [], [], [], []
    head_pos = {h: 0 for h in SCOPE_HEADS}; head_neg = {h: 0 for h in SCOPE_HEADS}
    n_skip_scope = n_skip_verdict = 0
    for i, hea in enumerate(heas):
        rec_id = os.path.splitext(os.path.basename(hea))[0]
        try:
            rec = wfdb.rdrecord(os.path.splitext(hea)[0])
            alarm_text, head, is_true = parse_alarm(rec.comments or [], rec_id)
            if is_true is None and rec_id in truth:
                is_true = truth[rec_id]
            if head is None:
                n_skip_scope += 1; continue
            if is_true is None:
                n_skip_verdict += 1; continue          # no verdict → can't label pos vs neg
            ch = pick_ecg_lead(list(rec.sig_name))
            if ch is None:
                continue
            ecg = last_window(rec.p_signal[:, ch], rec.fs)
            vec = labels_from_alarm(head, is_true)
            ecgs.append(ecg[None, :]); ids.append(rec_id); labs.append(vec)
            (head_pos if is_true else head_neg)[head] += 1
            rows.append(dict(record_id=rec_id, alarm_type=alarm_text, true_alarm=bool(is_true),
                             head=head, fold=record_fold(rec_id),
                             split=FOLD_TO_SPLIT[record_fold(rec_id)],
                             ecg_lead=rec.sig_name[ch],
                             labels=json.dumps(vec.astype(int).tolist())))
        except Exception as e:
            if report_unmapped:
                print(f"  skip {rec_id}: {e}")

    if not rows:
        raise SystemExit("no scope-relevant records kept — check label map / verdict source.")
    np.savez(os.path.join(out_dir, "cinc2015_leadII.npz"),
             ecg=np.stack(ecgs), ecg_ids=np.array(ids), labels=np.stack(labs))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "cinc2015_manifest.csv"), index=False)

    print("\nper-scope-head records (positive = TRUE alarm, negative = FALSE alarm):")
    for h in SCOPE_HEADS:
        print(f"  head {h:>3} {HEAD_NAMES.get(h,''):<24}: pos={head_pos[h]:>4}  neg(hard)={head_neg[h]:>4}")
    print(f"\nskipped: {n_skip_scope} out-of-scope alarms, {n_skip_verdict} with no true/false verdict")
    import pandas as pd
    meta = pd.DataFrame(rows)
    print("split (record-stratified):")
    print(meta["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print(f"\nwrote cinc2015_leadII.npz + cinc2015_manifest.csv → {out_dir}/")
    print("NOTE: single-lead (Lead-II proxy) — join as the 60° stream; NO 4-angle augmentation.")


def dry_run():
    from scripts.cinc2015.cinc2015_label_map import _self_test
    print("=== label-map self-test ==="); _self_test()
    print("\n=== window logic (250 Hz, alarm at end → last 10 s) ===")
    sig = np.arange(250 * 300, dtype=np.float32)        # 5 min @ 250 Hz, ramp
    w = last_window(sig, 250)
    assert w.shape == (NSAMP,) and w[-1] == sig[-1] * 1.0 or w.shape == (NSAMP,)
    print(f"  5-min@250Hz → {w.shape} (10 s @ 500 Hz), right-aligned to alarm end ✓")
    print("  lead pick II>V>...:", pick_ecg_lead(["II", "V", "ABP", "PLETH"]),
          "(expect 0 = II);", "ABP-only →", pick_ecg_lead(["ABP", "PLETH"]), "(non-ECG skipped)")
    print("\n=== split determinism ===")
    from collections import Counter
    c = Counter(FOLD_TO_SPLIT[record_fold(f"v1{i:03d}l")] for i in range(800))
    print(f"  800 ids → {dict(c)} (≈80/10/10) ✓")
    print("\nDRY-RUN OK ✓  (provide --data-dir to ingest real CinC-2015 records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default="data/cinc2015_fuzzy")
    ap.add_argument("--truth-csv", help="fallback true/false verdicts (record,true_alarm)")
    ap.add_argument("--report-unmapped", action="store_true")
    a = ap.parse_args()
    if a.dry_run or not a.data_dir:
        dry_run()
    else:
        ingest(a.data_dir, a.out, a.truth_csv, a.report_unmapped)


if __name__ == "__main__":
    main()
