"""Partial download of MIMIC-IV-ECG for fuzzy-fusion training.

Goal
----
Pull 10 000 positives + 10 000 negatives for EACH of the 9 unified Founder
heads (idx 4, 5, 6, 9, 16, 19, 93, 98, 142). Deduped across heads (a single
record can satisfy multiple heads), so the total record count is between
20 000 (perfect overlap) and 180 000 (no overlap). Empirically expected
around 100 000–130 000 unique records.

Pipeline (4 steps, all run via this script)
------------------------------------------
  1. Download metadata only:  machine_measurements.csv + record_list.csv
  2. Parse + select records:  apply scripts/mimic/mimic_label_map.py;
                              choose 10k pos + 10k neg per head with a
                              deterministic seed; emit selected_records.csv
                              (one row per chosen record).
  3. Download waveforms:      wget --user/--password against PhysioNet
                              for the selected (subject_id, study_id) only.
  4. Convert to 4-angle npz:  apply the V_θ = V_I cos θ + Y sin θ formula
                              at θ ∈ {45°, 60°, 75°, 90°} per record, save
                              data/mimic-iv-ecg-fuzzy/train_{angle}deg.npz
                              with shape (n, 1, 5000) float32 + binary
                              (n, 150) labels matching the existing
                              fuzzylead2 npz format so the existing
                              MultiAngleDataset loader accepts it.

Credentials
-----------
PhysioNet requires credentialed access (CITI training + signed MIMIC DUA).
Pass via env vars PHYSIONET_USER and PHYSIONET_PASS, OR via --user / --password
CLI flags, OR via interactive prompt.

Resumability
------------
All steps are idempotent. Step 1 uses wget -N -c. Step 3 skips records whose
.dat + .hea are already on disk. Step 4 re-uses cached selected_records.csv.

Disk budget
-----------
- Step 1 metadata: ~250 MB
- Step 3 waveforms: ~150 GB for ~125k records (each .dat is ~1.2 MB at 500 Hz)
- Step 4 npz output: 4 angles × ~125k × 1 × 5000 × 4 bytes ≈ 10 GB

Bandwidth budget
----------------
~150 GB over HTTPS at PhysioNet's typical 1–10 MB/s = several hours to a full day.
Use --aws-s3 to switch to the requester-pays S3 mirror (you pay egress).
"""
import argparse
import json
import os
import sys
import getpass
import subprocess
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.mimic.mimic_label_map import (
    HEAD_IDX, HEAD_NAMES, HEAD_SPEC,
    labels_from_report_columns, labels_to_founder_vector,
)

# ─── Constants ────────────────────────────────────────────────────────────
MIMIC_URL  = "https://physionet.org/files/mimic-iv-ecg/1.0/"
S3_BUCKET  = "s3://physionet-open/mimic-iv-ecg/1.0/"
DEST_ROOT  = Path("data/mimic-iv-ecg")
META_DIR   = DEST_ROOT / "metadata"
WAV_DIR    = DEST_ROOT / "files"
SELECT_CSV = DEST_ROOT / "selected_records.csv"
FUZZY_DIR  = Path("data/mimic-iv-ecg-fuzzy")
ANGLES_TO_BUILD = (45, 60, 75, 90)
TARGET_FS  = 500
TARGET_LEN = 5000  # 10 s

# Columns we expect in machine_measurements.csv (PhysioNet documents
# `report_0`..`report_17` plus `subject_id`, `study_id`, `cart_id`, `ecg_time`).
REPORT_COLS = [f"report_{i}" for i in range(18)]


# ─── Step 1: metadata download ────────────────────────────────────────────
def step1_download_metadata(user: str, pwd: str, force: bool = False) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    targets = ("machine_measurements.csv", "record_list.csv", "LICENSE.txt")
    for fn in targets:
        url   = urllib.parse.urljoin(MIMIC_URL, fn)
        out   = META_DIR / fn
        if out.exists() and not force:
            print(f"  [skip] {out} already present ({out.stat().st_size/1e6:.1f} MB)")
            continue
        cmd = ["wget", "-c", "-N", "--user", user, "--password", pwd, "-O", str(out), url]
        print(f"  $ wget {url}")
        subprocess.run(cmd, check=True)


# ─── Step 2: parse + select ────────────────────────────────────────────────
def step2_select_records(
    n_pos: int = 10_000, n_neg: int = 10_000, seed: int = 42,
) -> pd.DataFrame:
    print(f"  reading {META_DIR / 'machine_measurements.csv'} ...")
    mm = pd.read_csv(META_DIR / "machine_measurements.csv")
    have_cols = [c for c in REPORT_COLS if c in mm.columns]
    print(f"  found {len(mm):,} machine_measurements rows, "
          f"{len(have_cols)}/{len(REPORT_COLS)} report_* columns present")

    # Apply label_map row-by-row.
    rng = np.random.default_rng(seed)
    print(f"  applying 9-head label mapper ...")
    labels = np.zeros((len(mm), 9), dtype=np.int8)
    for i in tqdm(range(len(mm)), desc="  labeling", mininterval=2.0):
        hits = labels_from_report_columns(mm.iloc[i][have_cols].tolist())
        for j, idx in enumerate(HEAD_IDX):
            if idx in hits:
                labels[i, j] = 1

    # Per-head selection — 10k pos + 10k neg, dedup across heads at the end.
    selected: set[int] = set()
    per_head_summary = []
    for j, head_idx in enumerate(HEAD_IDX):
        pos_pool = np.where(labels[:, j] == 1)[0]
        neg_pool = np.where(labels[:, j] == 0)[0]
        if len(pos_pool) < n_pos:
            print(f"  ⚠ idx {head_idx}: only {len(pos_pool)} positives available "
                  f"(asked for {n_pos}); taking all.")
        n_take_pos = min(n_pos, len(pos_pool))
        n_take_neg = min(n_neg, len(neg_pool))
        pos_sample = rng.choice(pos_pool, size=n_take_pos, replace=False)
        neg_sample = rng.choice(neg_pool, size=n_take_neg, replace=False)
        selected.update(pos_sample.tolist())
        selected.update(neg_sample.tolist())
        per_head_summary.append({
            "head_idx": head_idx, "head_name": HEAD_NAMES[j],
            "positives_available": int(len(pos_pool)),
            "positives_selected":  int(n_take_pos),
            "negatives_available": int(len(neg_pool)),
            "negatives_selected":  int(n_take_neg),
        })
    print("\n  per-head selection summary:")
    print(pd.DataFrame(per_head_summary).to_string(index=False))
    print(f"\n  unique records selected: {len(selected):,} "
          f"(theoretical max with no overlap: {9 * (n_pos + n_neg):,})")

    sel_idx = sorted(selected)
    df_sel = mm.iloc[sel_idx].copy()
    df_sel["__row_idx"] = sel_idx
    # Persist the 9-head label vector + full-150 vector inline for convenience
    df_sel["labels_9_binary"]   = [labels[i].tolist() for i in sel_idx]
    df_sel["labels_150_binary"] = [
        labels_to_founder_vector(
            [HEAD_IDX[j] for j, v in enumerate(labels[i]) if v]
        ) for i in sel_idx
    ]
    df_sel.to_csv(SELECT_CSV, index=False)
    print(f"\n  selected records → {SELECT_CSV}")
    return df_sel


# ─── Step 3: download waveforms for selected records ──────────────────────
def _record_path(subject_id: int, study_id: int) -> str:
    """Return PhysioNet relative path for a (subject_id, study_id) waveform.

    MIMIC-IV-ECG layout:
      files/p{subject_id // 1000_000}/p{subject_id}/s{study_id}/{study_id}.dat
    where the first directory uses the millions bucket. Tweak this if the
    real layout differs after step 1 metadata reveals it.
    """
    p_bucket = subject_id // 1_000_000
    return (f"files/p{p_bucket}/p{subject_id}/s{study_id}/{study_id}")


def step3_download_waveforms(
    df_sel: pd.DataFrame, user: str, pwd: str, use_s3: bool = False, parallel: int = 4,
) -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    paths = [_record_path(int(r["subject_id"]), int(r["study_id"]))
             for _, r in df_sel.iterrows()]
    print(f"  downloading {len(paths):,} records ({'AWS S3' if use_s3 else 'PhysioNet HTTPS'}) ...")
    if use_s3:
        for p in tqdm(paths, desc="  s3 sync", mininterval=2.0):
            for ext in (".dat", ".hea"):
                src = f"{S3_BUCKET}{p}{ext}"
                dst = DEST_ROOT / f"{p}{ext}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists(): continue
                subprocess.run(
                    ["aws", "s3", "cp", "--request-payer", "requester", src, str(dst)],
                    check=True,
                )
    else:
        for p in tqdm(paths, desc="  wget", mininterval=2.0):
            for ext in (".dat", ".hea"):
                url = urllib.parse.urljoin(MIMIC_URL, f"{p}{ext}")
                dst = DEST_ROOT / f"{p}{ext}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists(): continue
                subprocess.run(
                    ["wget", "-q", "-c", "-N", "--user", user, "--password", pwd,
                     "-O", str(dst), url], check=True,
                )


# ─── Step 4: convert to 4-angle npz ───────────────────────────────────────
def step4_build_fuzzy_npz(df_sel: pd.DataFrame) -> None:
    import wfdb
    FUZZY_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df_sel)
    labels_150 = np.asarray(df_sel["labels_150_binary"].apply(eval).tolist(),
                             dtype=np.float32)
    print(f"  building {len(ANGLES_TO_BUILD)} npz files at angles {ANGLES_TO_BUILD} "
          f"for {n:,} records ...")
    # Pre-allocate per-angle ecg arrays
    ecg_by_angle = {a: np.zeros((n, 1, TARGET_LEN), dtype=np.float32) for a in ANGLES_TO_BUILD}
    failed = []
    for i, (_, row) in enumerate(tqdm(list(df_sel.iterrows()), desc="  derive 4 angles", mininterval=2.0)):
        rec_path = DEST_ROOT / _record_path(int(row["subject_id"]), int(row["study_id"]))
        try:
            rec = wfdb.rdrecord(str(rec_path))
            assert rec.fs == TARGET_FS, f"unexpected fs={rec.fs}"
            # Lead I = sig index 0, Lead II = sig index 1 in MIMIC-IV-ECG
            li  = rec.p_signal[:, 0].astype(np.float32)
            lii = rec.p_signal[:, 1].astype(np.float32)
            if li.shape[0] < TARGET_LEN:
                li  = np.pad(li,  (0, TARGET_LEN - li.shape[0]))
                lii = np.pad(lii, (0, TARGET_LEN - lii.shape[0]))
            li, lii = li[:TARGET_LEN], lii[:TARGET_LEN]
            y = (2 * lii - li) / np.sqrt(3)   # the "Y" orthogonal component
            for a in ANGLES_TO_BUILD:
                if a == 60:
                    ecg_by_angle[a][i, 0, :] = lii  # 60° == Lead II exactly
                else:
                    theta = np.deg2rad(a)
                    ecg_by_angle[a][i, 0, :] = li * np.cos(theta) + y * np.sin(theta)
        except Exception as exc:
            failed.append((i, str(rec_path), repr(exc)))

    if failed:
        print(f"  ⚠ {len(failed)} records failed; saving failure log to {FUZZY_DIR / 'failures.log'}")
        (FUZZY_DIR / "failures.log").write_text(
            "\n".join(f"{i}\t{path}\t{exc}" for i, path, exc in failed)
        )

    # Write per-angle npz
    ecg_ids = df_sel["__row_idx"].to_numpy(dtype=np.int32)
    for a in ANGLES_TO_BUILD:
        out = FUZZY_DIR / f"train_{a}deg.npz"
        np.savez(out, ecg=ecg_by_angle[a], labels=labels_150, ecg_ids=ecg_ids)
        sz = out.stat().st_size / 1e6
        print(f"  wrote {out}  ({sz:.1f} MB, shape={ecg_by_angle[a].shape})")
    # Also stash the 9-head binary matrix for convenience
    labels_9 = np.asarray(df_sel["labels_9_binary"].apply(eval).tolist(), dtype=np.int8)
    np.save(FUZZY_DIR / "labels_9_binary.npy", labels_9)
    print(f"  wrote {FUZZY_DIR / 'labels_9_binary.npy'}  shape={labels_9.shape}")


# ─── Entry ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--user",     default=os.environ.get("PHYSIONET_USER"))
    ap.add_argument("--password", default=os.environ.get("PHYSIONET_PASS"))
    ap.add_argument("--n-pos",    type=int, default=10_000)
    ap.add_argument("--n-neg",    type=int, default=10_000)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--aws-s3",   action="store_true",
                    help="Use the s3://physionet-open requester-pays mirror "
                         "instead of HTTPS (much faster if AWS CLI is configured).")
    ap.add_argument("--steps",    default="1,2,3,4",
                    help="Comma-separated subset of steps to run. Default: all.")
    ap.add_argument("--force-meta", action="store_true",
                    help="Re-download metadata even if present.")
    args = ap.parse_args()

    # Prompt for credentials if missing
    if not args.user:
        args.user = input("PhysioNet username: ").strip()
    if not args.password:
        args.password = getpass.getpass("PhysioNet password: ")

    steps = set(args.steps.split(","))
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    if "1" in steps:
        print("\n═══ Step 1: download metadata ═══")
        step1_download_metadata(args.user, args.password, force=args.force_meta)

    if "2" in steps:
        print("\n═══ Step 2: parse + select 10k pos + 10k neg per head ═══")
        df_sel = step2_select_records(args.n_pos, args.n_neg, args.seed)
    elif "3" in steps or "4" in steps:
        print(f"  loading cached selection from {SELECT_CSV} ...")
        df_sel = pd.read_csv(SELECT_CSV)
    else:
        df_sel = None

    if "3" in steps:
        print("\n═══ Step 3: download waveforms ═══")
        step3_download_waveforms(df_sel, args.user, args.password, use_s3=args.aws_s3)

    if "4" in steps:
        print("\n═══ Step 4: build 4-angle npz files ═══")
        step4_build_fuzzy_npz(df_sel)

    print("\nDone.")


if __name__ == "__main__":
    main()
