"""Compare base vs fuzzy-fine-tuned ECGFounder on the fzark TP cohort.

Re-runs inference with the fine-tuned checkpoint on the same 4,088-record
fzark stratified sample (random_state=42, 500/class) and reports per-V3.1
class detection at t=0.5 alongside the cached base-model numbers.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from label_config import FZARK_LABEL_MAP
from preprocessing import ECGPreprocessor

DATA_DIR     = "data/ecg_tp_fzark"
BASE_PROBS   = "res/tp_fzark_full_suppression/baseline_probs_tp.npy"
import argparse
FUZZY_CKPT_DEFAULT = "checkpoint/1_lead_ECGFounder_fuzzy.pth"
OUT_DIR            = "res/finetune_fuzzylead2"


class TpDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.prep = ECGPreprocessor.for_fzark()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        rel = str(self.df.iloc[idx]['JSON File']).replace('\\', '/')
        path = os.path.join(self.data_dir, rel)
        return self.prep.from_fzark_json(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=FUZZY_CKPT_DEFAULT)
    ap.add_argument('--label', default='fuzzy', help="Short label for output files (e.g. 'fuzzy' or 'fuzzy_masked').")
    args = ap.parse_args()
    fuzzy_ckpt = args.ckpt
    out_label  = args.label

    device = resolve_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {device}")
    df = pd.read_csv(f"{DATA_DIR}/summary.csv")
    sampled = [g.sample(n=min(500, len(g)), random_state=42)
               for _, g in df.groupby('Event Type')]
    df = pd.concat(sampled).reset_index(drop=True)
    print(f"Cohort: {len(df)} records")

    ds = TpDataset(df, DATA_DIR)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Loading fine-tuned model from {fuzzy_ckpt} ...")
    model = load_ecgfounder(device, ckpt_path=fuzzy_ckpt)
    model.eval()

    print("Running inference (fuzzy fine-tuned)...")
    all_probs = []
    with torch.no_grad():
        for x in tqdm(dl, desc="  fuzzy fwd"):
            x = x.to(device)
            logits = model(x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    fuzzy_probs = np.concatenate(all_probs, axis=0)
    np.save(os.path.join(OUT_DIR, f"{out_label}_probs_tp.npy"), fuzzy_probs)
    print(f"  shape: {fuzzy_probs.shape}")

    base_probs = np.load(BASE_PROBS)
    assert base_probs.shape == fuzzy_probs.shape

    # Per-V3.1 class side-by-side
    rows = []
    print(f"\n{'Event':<35} {'idx':>4} {'n':>4}  {'BASE det/mean':>16}  {'FUZZY det/mean':>16}  {'Δ det':>7}")
    print("-" * 100)
    for et, idx in sorted(FZARK_LABEL_MAP.items()):
        mask = (df['Event Type'] == et).values
        n = int(mask.sum())
        if n == 0:
            continue
        b_p, f_p = base_probs[mask, idx], fuzzy_probs[mask, idx]
        b_det = float((b_p >= 0.5).mean())
        f_det = float((f_p >= 0.5).mean())
        b_mean = float(b_p.mean()); f_mean = float(f_p.mean())
        rows.append(dict(event_type=et, head_idx=idx, n=n,
                         base_mean=b_mean, base_det50=b_det,
                         fuzzy_mean=f_mean, fuzzy_det50=f_det,
                         delta_det=f_det - b_det))
        print(f"{et:<35} {idx:>4} {n:>4}  {100*b_det:>6.1f}% / {b_mean:>5.3f}  {100*f_det:>6.1f}% / {f_mean:>5.3f}  {100*(f_det-b_det):>+6.1f}pp")

    out_csv = os.path.join(OUT_DIR, f"fzark_base_vs_{out_label}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nCSV → {out_csv}")

    # Aggregate
    print(f"\n{'Aggregate (V3.1, weighted by n)':<45} {'BASE':>10} {'FUZZY':>10} {'Δ':>8}")
    N = sum(r['n'] for r in rows)
    for thr in [0.5, 0.6, 0.7]:
        b = sum(int((base_probs[(df['Event Type']==r['event_type']).values, r['head_idx']] >= thr).sum()) for r in rows)
        f = sum(int((fuzzy_probs[(df['Event Type']==r['event_type']).values, r['head_idx']] >= thr).sum()) for r in rows)
        print(f"  detection @ t={thr:.1f}{'':<33} {100*b/N:>8.1f}% {100*f/N:>9.1f}% {100*(f-b)/N:>+7.1f}pp")


if __name__ == '__main__':
    main()
