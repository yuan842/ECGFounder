"""
eval_ecg_tprex.py

End-to-end evaluation of the single-lead ECGFounder model on the
ecg-tp_rex private annotated dataset (validation split).

Pre-processing pipeline per segment:
  1. Parse JSON frames → concatenate ECG integer samples
  2. Divide by magnification (1000) → convert to mV
  3. 50 Hz European notch filter (Belgian Holter device)
  4. 0.67–40 Hz Butterworth bandpass filter
  5. Median baseline-wander subtraction
  6. Resample: 128 Hz → 500 Hz (linear interpolation)
  7. Center-crop to exactly 5000 samples (10 s)
  8. Robust Winsorized Z-score normalization (1.5 / 98.5 percentile clipping)

Label mapping (ecg-tp_rex → ECGFounder tasks.txt index):
  Atrial Fibrillation            → 5
  Isolated Ventricular Beat      → 9  (PREMATURE VENTRICULAR COMPLEXES)
  Prolonged RR Interval          → 81 (PROLONGED QT – closest proxy)
  Ventricular Couplet            → 91 (PREMATURE VENTRICULAR AND FUSION COMPLEXES)
  Ventricular Run                → 99 (VENTRICULAR TACHYCARDIA)
  Pause                          → 143 (WITH SINUS PAUSE)
  Sinus Tachycardia              → 6
  Supraventricular Couplet       → 20 (PREMATURE SUPRAVENTRICULAR COMPLEXES)
  Isolated Supraventricular Beat → 16 (PREMATURE ATRIAL COMPLEXES)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from device_utils import resolve_device
from preprocessing import ECGPreprocessor, TARGET_LEN
from checkpoints import load_ecgfounder
from label_config import FZARK_LABEL_MAP as LABEL_MAP

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = "./data/ecg-tp_rex"
VAL_CSV       = "./data/ecg-tp_rex/splits/val.csv"
CKPT_PATH     = "./checkpoint/1_lead_ECGFounder.pth"
SAVED_DIR     = "./res/tprex_eval"
os.makedirs(SAVED_DIR, exist_ok=True)

BATCH_SIZE    = 64

# LABEL_MAP is imported from `label_config` (v3 single-index ontology).
# It supersedes the pre-cleanup map with five corrected/added indices and
# six dropped composite events (Ventricular Couplet, Prolonged RR Interval,
# bigeminy/trigeminy events) which now pass through unmapped.

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TpRex_SingleLead_Dataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str):
        self.df       = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.prep     = ECGPreprocessor.for_fzark()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        json_rel = row['JSON File']
        event    = row['Event Type']

        json_path = os.path.join(self.data_dir, json_rel)

        # Build 150-class label vector
        label = np.zeros(150, dtype=np.float32)
        if event in LABEL_MAP:
            label[LABEL_MAP[event]] = 1.0

        try:
            tensor = self.prep.from_fzark_json(json_path)
        except Exception as e:
            # Return zeros on parse failure — logged downstream
            print(f"[WARN] Failed to parse {json_path}: {e}")
            tensor = torch.zeros(1, TARGET_LEN)

        return tensor, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference"):
            probs = torch.sigmoid(model(x.to(device)))
            all_preds.append(probs.cpu().numpy())
            all_gts.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_gts)


def main():
    print("=" * 65)
    print("ecg-tp_rex  |  Single-Lead ECGFounder Evaluation")
    print("Pre-processing: Robust Winsorized Z-score  |  50 Hz Notch")
    print("=" * 65)

    device = resolve_device()        # project default: MPS → CUDA → CPU
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    val_ds  = TpRex_SingleLead_Dataset(VAL_CSV, DATA_DIR)
    val_dl  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Validation segments: {len(val_ds)}")

    # Print val class breakdown
    df_val = pd.read_csv(VAL_CSV)
    print("\nValidation class distribution:")
    for cls, cnt in df_val['Event Type'].value_counts().items():
        idx = LABEL_MAP.get(cls, '?')
        print(f"  [{idx:>3}] {cls:<35} {cnt}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = load_ecgfounder(device, ckpt_path=CKPT_PATH)
    print("Model ready.")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_gts = run_inference(model, val_dl, device)
    np.save(os.path.join(SAVED_DIR, "preds.npy"), all_preds)
    np.save(os.path.join(SAVED_DIR, "gts.npy"),   all_gts)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Performance Metrics  (val n={len(val_ds)})")
    print(f"{'='*65}")
    header = f"{'Event Type':<36} {'Pos':>5}  {'ROC-AUC':>7}  {'PR-AUC':>7}  {'F1@0.5':>7}"
    print(header)
    print("-" * 65)

    results = []
    for event, cls_idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        y_true = all_gts[:, cls_idx]
        y_pred = all_preds[:, cls_idx]
        pos    = int(y_true.sum())

        if pos > 0 and pos < len(y_true):
            roc = roc_auc_score(y_true, y_pred)
            pr  = average_precision_score(y_true, y_pred)
            f1  = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
            print(f"{event:<36} {pos:>5}  {roc:>7.4f}  {pr:>7.4f}  {f1:>7.4f}")
            results.append(dict(event_type=event, class_index=cls_idx,
                                positives=pos, roc_auc=roc, pr_auc=pr, f1_score=f1))
        else:
            print(f"{event:<36} {pos:>5}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")

    print("=" * 65)

    # Macro-average over evaluated classes
    if results:
        df_res = pd.DataFrame(results)
        print(f"\nMacro-average (over {len(results)} evaluable classes):")
        print(f"  ROC-AUC : {df_res['roc_auc'].mean():.4f}")
        print(f"  PR-AUC  : {df_res['pr_auc'].mean():.4f}")
        print(f"  F1@0.5  : {df_res['f1_score'].mean():.4f}")

        out_csv = os.path.join(SAVED_DIR, "tprex_singlelead_metrics.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")

    # Note: v2 feature-gate FP suppression is applied separately via
    # `compare_tp_fzark_with_full_suppression.py`. This script only produces
    # raw model probabilities.

if __name__ == "__main__":
    main()
