"""
mitdb_eval.py

Evaluates the single-lead (Lead II) ECGFounder model on MIT-BIH validation split
under two pre-processing conditions (both via `preprocessing.ECGPreprocessor`):
  1. Standard Z-Score   (normalize="zscore")
  2. Robust Winsorized  (normalize="winsorize", default for `for_mitdb()`)

Outputs a 2-way comparison table:
  1-Lead Standard | 1-Lead Robust
"""

import os
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from device_utils import resolve_device
from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from label_config import (
    MITDB_BEAT_MAP,
    MITDB_RHYTHM_MAP,
    MITDB_DEFAULT_NORMAL,
)


# ─── Shared label extractor ──────────────────────────────────────────────────

def get_segment_labels(record_path, segment_index, fs_mitdb=360):
    """Build a 150-element binary label vector for one MIT-BIH segment.

    Beat and rhythm dictionaries are imported from `label_config` (v3 ontology).
    The S and j beat symbols route to idx 19 (PSVC) per v3, distinct from
    A/a which route to idx 16 (PAC).
    """
    ann = wfdb.rdann(record_path, 'atr')
    start_sample = segment_index * (10 * fs_mitdb)
    end_sample   = start_sample + (10 * fs_mitdb)

    idxs = np.where((ann.sample >= start_sample) & (ann.sample < end_sample))[0]
    segment_symbols = [ann.symbol[i] for i in idxs]

    labels = np.zeros(150, dtype=np.float32)

    # Identify the rhythm active at the segment start (carry-forward).
    active_rhythm = '(N'
    for s, note in zip(ann.sample, ann.aux_note):
        if s <= start_sample:
            if note != '':
                active_rhythm = note
        else:
            break
    segment_notes = [ann.aux_note[i] for i in idxs if ann.aux_note[i] != '']

    # Rhythm-level mapping: AFIB / VT / AFL / SVTA / SBR
    has_abnormal_rhythm = False
    for rhythm_token, idx in MITDB_RHYTHM_MAP.items():
        active = rhythm_token in active_rhythm
        seen   = any(rhythm_token in n for n in segment_notes)
        if active or seen:
            labels[idx] = 1.0
            has_abnormal_rhythm = True

    # Beat-level mapping: V / L / R / A / a / S / j
    has_abnormal_beat = False
    for sym in segment_symbols:
        idx = MITDB_BEAT_MAP.get(sym)
        if idx is not None:
            labels[idx] = 1.0
            has_abnormal_beat = True

    # Default-normal assignment when neither rhythm nor beat abnormalities present.
    if not has_abnormal_rhythm and not has_abnormal_beat:
        for idx in MITDB_DEFAULT_NORMAL:
            labels[idx] = 1.0

    return labels


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MITDB_SingleLead_Dataset(Dataset):
    """
    Loads raw MIT-BIH segment windows on-the-fly and runs the unified
    preprocessing pipeline configured for MIT-BIH (60 Hz powerline, MLII lead).
    Outputs a (1, 5000) single-lead tensor.
    """
    def __init__(self, metadata_path, mitdb_dir, prep, target_length=5000, fs_mitdb=360):
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df['split'] == 'val'].reset_index(drop=True)
        self.mitdb_dir     = mitdb_dir
        self.prep          = prep
        self.target_length = target_length
        self.fs_mitdb      = fs_mitdb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        record  = str(row['parent_record'])
        seg_idx = int(row['segment_index'])

        record_path  = os.path.join(self.mitdb_dir, record)
        start_sample = seg_idx * (10 * self.fs_mitdb)
        end_sample   = start_sample + (10 * self.fs_mitdb)

        # Load raw window
        signals, fields = wfdb.rdsamp(record_path,
                                      sampfrom=start_sample,
                                      sampto=end_sample)
        source_leads = fields['sig_name']
        signals = np.transpose(signals, (1, 0))   # (channels, samples)

        # Run the configured preprocessing pipeline
        std_tensor = self.prep.process(
            signal=signals,
            fs_in=self.fs_mitdb,
            source_leads=source_leads,
        )

        label = get_segment_labels(record_path, seg_idx, self.fs_mitdb)
        return std_tensor, torch.tensor(label, dtype=torch.float32)


# ─── Evaluation helper ────────────────────────────────────────────────────────

TARGET_CLASSES = {
    1:  "NORMAL SINUS RHYTHM",
    4:  "SINUS BRADYCARDIA",                          # v3: from (SBR rhythm
    5:  "ATRIAL FIBRILLATION",
    9:  "PREMATURE VENTRICULAR COMPLEXES",
    11: "RIGHT BUNDLE BRANCH BLOCK",
    16: "PREMATURE ATRIAL COMPLEXES",
    19: "PREMATURE SUPRAVENTRICULAR COMPLEXES",       # v3: from S/j beats
    20: "LEFT BUNDLE BRANCH BLOCK",
    32: "ATRIAL FLUTTER",                             # v3: from (AFL rhythm
    93: "SUPRAVENTRICULAR TACHYCARDIA",               # v3: from (SVTA rhythm
    98: "VENTRICULAR TACHYCARDIA",                    # v3: from (VT rhythm
}

def run_inference(model, loader, device):
    all_preds, all_gts = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            probs  = torch.sigmoid(model(inputs))
            all_preds.append(probs.cpu().numpy())
            all_gts.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_gts)


def compute_metrics(all_preds, all_gts):
    results = {}
    for idx, name in TARGET_CLASSES.items():
        y_true = all_gts[:, idx]
        y_pred = all_preds[:, idx]
        pos    = int(y_true.sum())
        if pos > 0 and pos < len(y_true):
            roc = roc_auc_score(y_true, y_pred)
            pr  = average_precision_score(y_true, y_pred)
            f1  = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
        else:
            roc = pr = f1 = None
        results[idx] = dict(name=name, pos=pos, roc=roc, pr=pr, f1=f1)
    return results




# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    metadata_path  = "./res/mitdb_standardized/mitdb_split_metadata.csv"
    mitdb_dir      = "./data/mitdb"
    ckpt_1lead     = "./checkpoint/1_lead_ECGFounder.pth"
    saved_dir      = "./res/mitdb_singlelead"
    os.makedirs(saved_dir, exist_ok=True)

    device = resolve_device()        # project default: MPS → CUDA → CPU
    print(f"Device: {device}\n")

    # ── Preprocessors (standard vs robust normalization) ──────────────────────
    std_prep = ECGPreprocessor(powerline_hz=60, target_lead="II", normalize="zscore")
    rob_prep = ECGPreprocessor.for_mitdb()    # winsorized z-score

    # ── 1-Lead Model ──────────────────────────────────────────────────────────
    print("Loading single-lead (Lead II) model …")
    model_1 = load_ecgfounder(device, ckpt_path=ckpt_1lead)

    # ── Standard preprocessing → single-lead inference ────────────────────────
    print("Running: Single-Lead + Standard Z-Score preprocessing …")
    ds_std  = MITDB_SingleLead_Dataset(metadata_path, mitdb_dir, std_prep)
    dl_std  = DataLoader(ds_std, batch_size=128, shuffle=False, num_workers=0)
    preds_std, gts_std  = run_inference(model_1, dl_std, device)
    metrics_1std = compute_metrics(preds_std, gts_std)
    np.save(os.path.join(saved_dir, "preds_1lead_standard.npy"), preds_std)

    # ── Robust preprocessing → single-lead inference ──────────────────────────
    print("Running: Single-Lead + Robust Winsorized preprocessing …")
    ds_rob  = MITDB_SingleLead_Dataset(metadata_path, mitdb_dir, rob_prep)
    dl_rob  = DataLoader(ds_rob, batch_size=128, shuffle=False, num_workers=0)
    preds_rob, gts_rob  = run_inference(model_1, dl_rob, device)
    metrics_1rob = compute_metrics(preds_rob, gts_rob)
    np.save(os.path.join(saved_dir, "preds_1lead_robust.npy"), preds_rob)
    np.save(os.path.join(saved_dir, "gts.npy"),                gts_rob)

    # ── Print 1-lead Standard vs Robust comparison ────────────────────────────
    hdr = (f"\n{'═'*82}\n"
           f"{'':32s}  {'1-Lead Standard':^18s}  {'1-Lead Robust':^18s}\n"
           f"{'Class':32s}  {'ROC / PR / F1':^18s}  {'ROC / PR / F1':^18s}\n"
           f"{'─'*82}")
    print(hdr)

    results_rows = []
    for idx, info in metrics_1std.items():
        name = info['name']
        pos  = info['pos']

        def fmt1(m):
            if m['roc'] is None:
                return "  N/A  / N/A  / N/A "
            return f"{m['roc']:.3f}/{m['pr']:.3f}/{m['f1']:.3f}"

        col_1std  = fmt1(info)
        col_1rob  = fmt1(metrics_1rob[idx])

        print(f"{name:<32s}  {col_1std:^18s}  {col_1rob:^18s}")

        results_rows.append({
            "class_index": idx, "class_name": name, "positives": pos,
            "1lead_std_roc":  info['roc'],
            "1lead_std_pr":   info['pr'],
            "1lead_std_f1":   info['f1'],
            "1lead_rob_roc":  metrics_1rob[idx]['roc'],
            "1lead_rob_pr":   metrics_1rob[idx]['pr'],
            "1lead_rob_f1":   metrics_1rob[idx]['f1'],
        })

    print(f"{'═'*82}")

    # Save full comparison CSV
    out_csv = os.path.join(saved_dir, "singlelead_comparison.csv")
    pd.DataFrame(results_rows).to_csv(out_csv, index=False)
    print(f"\nFull comparison saved to: '{out_csv}'")


if __name__ == "__main__":
    main()
