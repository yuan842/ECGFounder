"""Ingest Chapman-Shaoxing / Ningbo (PhysioNet 12-lead) → fuzzy-build format.

Open-access (NO DUA). Drop-in counterpart to the PTB-XL fuzzy dataset: reads each
12-lead WFDB record, derives the 4 synthetic lead angles {45,60,75,90°} (same
frontal-plane projection as data/fuzzylead2/derivelead2.md), maps SNOMED-CT → founder
heads, and writes per-angle npz + a record-stratified split manifest (no record/patient
spans train/val/test, mirroring the global PTB-XL 1-8/9/10 rule).

Primary goal: top up SVT (head 93). Also yields Normal/Brady/AFib/SinusTachy. VT(98) is
sparse and Pause(142) ~absent in this corpus (see chapman_label_map) — use CinC-2015 for those.

Usage:
  python scripts/chapman/ingest_chapman.py --dry-run                 # validate, no data needed
  python scripts/chapman/ingest_chapman.py --data-dir data/chapman_ningbo \
         --out data/chapman_fuzzy [--n-per-head 900] [--report-unmapped]

Expected --data-dir layout: WFDB records (*.hea + *.mat/*.dat), recursively. Each .hea
must carry a `# Dx: <snomed,...>` line and be 12-lead. Records are resampled to 500 Hz /
5000 samples (10 s) if needed.
"""
from __future__ import annotations
import os, sys, glob, json, argparse, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.chapman.chapman_label_map import labels_from_dx, HEAD_NAMES, SCOPE_HEADS

ANGLES = (45, 60, 75, 90)
FS, NSAMP = 500, 5000
LEAD_I, LEAD_II = "I", "II"
# record-stratified split mirroring PTB-XL folds 1-8 / 9 / 10 (deterministic by record id)
FOLD_TO_SPLIT = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}


def derive_angles(lead_i: np.ndarray, lead_ii: np.ndarray) -> dict[int, np.ndarray]:
    """Frontal-plane projection from leads I & II (derivelead2.md). 60° == lead II."""
    y = (2.0 * lead_ii - lead_i) / np.sqrt(3.0)          # orthogonal (≈ aVF direction)
    out = {}
    for a in ANGLES:
        if a == 60:
            out[a] = lead_ii.astype(np.float32)
        else:
            th = np.deg2rad(a)
            out[a] = (lead_i * np.cos(th) + y * np.sin(th)).astype(np.float32)
    return out


def record_fold(record_id: str) -> int:
    """Deterministic 1..10 fold from a record id (stable across runs, no RNG).

    NOTE: Chapman/Ningbo are ~1 record per patient, so record-level == patient-level
    here. If a patient↔records table is provided, fold on patient_id instead to be safe.
    """
    h = int(hashlib.sha1(record_id.encode()).hexdigest(), 16)
    return (h % 10) + 1


def _resample_crop(sig: np.ndarray, fs_in: int) -> np.ndarray:
    """1-D → 500 Hz, 5000 samples (center-crop / zero-pad). Minimal, no filtering."""
    if fs_in != FS:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(fs_in), FS)
        sig = resample_poly(sig, FS // g, int(fs_in) // g)
    if len(sig) >= NSAMP:
        s = (len(sig) - NSAMP) // 2
        return sig[s:s + NSAMP].astype(np.float32)
    out = np.zeros(NSAMP, np.float32); out[:len(sig)] = sig
    return out


def ingest(data_dir: str, out_dir: str, n_per_head: int | None, report_unmapped: bool):
    import wfdb
    os.makedirs(out_dir, exist_ok=True)
    heas = sorted(glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True))
    if not heas:
        raise SystemExit(f"no .hea records under {data_dir}")
    print(f"found {len(heas)} records under {data_dir}")

    rows, ecg_by_angle, unmapped_counter = [], {a: [] for a in ANGLES}, {}
    head_counts = {h: 0 for h in SCOPE_HEADS}
    rng = np.random.default_rng(42)
    for i, hea in enumerate(heas):
        rec_id = os.path.splitext(os.path.basename(hea))[0]
        try:
            rec = wfdb.rdrecord(os.path.splitext(hea)[0])
            names = list(rec.sig_name)
            if LEAD_I not in names or LEAD_II not in names:
                continue
            dx = next((c for c in (rec.comments or []) if "Dx" in c), "")
            vec, mapped, unmapped = labels_from_dx(dx)
            for c in unmapped:
                unmapped_counter[c] = unmapped_counter.get(c, 0) + 1
            # keep only records that hit at least one scope head (optional cap per head)
            scope_hit = [h for h in mapped if h in SCOPE_HEADS]
            if not scope_hit:
                continue
            if n_per_head and all(head_counts[h] >= n_per_head for h in scope_hit):
                continue
            li = _resample_crop(rec.p_signal[:, names.index(LEAD_I)], rec.fs)
            lii = _resample_crop(rec.p_signal[:, names.index(LEAD_II)], rec.fs)
            ang = derive_angles(li, lii)
            for a in ANGLES:
                ecg_by_angle[a].append(ang[a])
            for h in scope_hit:
                head_counts[h] += 1
            rows.append(dict(record_id=rec_id, fold=record_fold(rec_id),
                             split=FOLD_TO_SPLIT[record_fold(rec_id)],
                             labels=json.dumps(vec.astype(int).tolist()),
                             scope_heads="+".join(map(str, scope_hit))))
        except Exception as e:
            if report_unmapped:
                print(f"  skip {rec_id}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(heas)}  kept={len(rows)}")

    if not rows:
        raise SystemExit("no scope-relevant records kept — check the label map / data.")
    import pandas as pd
    meta = pd.DataFrame(rows)
    eid = np.array([r["record_id"] for r in rows])
    labels = np.array([json.loads(r["labels"]) for r in rows], dtype=np.float32)
    for a in ANGLES:
        np.savez(os.path.join(out_dir, f"train_{a}deg.npz"),
                 ecg=np.stack(ecg_by_angle[a])[:, None, :], ecg_ids=eid, labels=labels)
    meta.to_csv(os.path.join(out_dir, "chapman_split_manifest.csv"), index=False)

    print("\nper-scope-head record counts:")
    for h in SCOPE_HEADS:
        print(f"  head {h:>3} {HEAD_NAMES.get(h,''):<28}: {head_counts[h]}")
    print("\nsplit (record-stratified, no leakage):")
    print(meta["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    if report_unmapped:
        top = sorted(unmapped_counter.items(), key=lambda x: -x[1])[:25]
        print("\ntop unmapped SNOMED codes (extend chapman_label_map if relevant):")
        for c, n in top:
            print(f"  {c}: {n}")
    print(f"\nwrote per-angle npz + chapman_split_manifest.csv → {out_dir}/")


def dry_run():
    """No data needed — validate label map + angle derivation + split determinism."""
    from scripts.chapman.chapman_label_map import _self_test
    print("=== label-map self-test ==="); _self_test()
    print("\n=== angle derivation sanity ===")
    t = np.linspace(0, 1, NSAMP, dtype=np.float32)
    li, lii = np.sin(2*np.pi*1.2*t), 1.3*np.sin(2*np.pi*1.2*t + 0.4)
    ang = derive_angles(li, lii)
    assert np.allclose(ang[60], lii), "60° must equal lead II"
    print(f"  angles built: {sorted(ang)}; shapes {ang[45].shape}; 60°==leadII ✓")
    print("\n=== split determinism / coverage ===")
    folds = [record_fold(f"JS{i:05d}") for i in range(3000)]
    from collections import Counter
    c = Counter(FOLD_TO_SPLIT[f] for f in folds)
    print(f"  3000 synthetic ids → {dict(c)} (≈80/10/10); deterministic, hash-based ✓")
    print("\nDRY-RUN OK ✓  (provide --data-dir to ingest real Chapman/Ningbo records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default="data/chapman_fuzzy")
    ap.add_argument("--n-per-head", type=int, default=None,
                    help="optional cap on kept records per scope head (e.g. 900)")
    ap.add_argument("--report-unmapped", action="store_true",
                    help="log SNOMED codes with no head mapping (audit coverage)")
    a = ap.parse_args()
    if a.dry_run or not a.data_dir:
        dry_run()
    else:
        ingest(a.data_dir, a.out, a.n_per_head, a.report_unmapped)


if __name__ == "__main__":
    main()
