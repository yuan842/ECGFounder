"""Verify DualHeadECGFounder on val_75deg + fzark cohorts.

Checks:
1. val_75deg active heads should match the FUZZY checkpoint exactly (when
   routed through fuzzy) and BASE exactly (when routed through base).
2. fzark V3.1 heads should match BASE on every head NOT routed to fuzzy
   (bit-identical). Heads routed to fuzzy will reflect fuzzy behavior.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from device_utils import resolve_device
from label_config import FZARK_LABEL_MAP
from preprocessing import ECGPreprocessor
from dual_head_ecgfounder import DualHeadECGFounder


def main():
    device = resolve_device()
    print(f"Device: {device}")

    # ── 1. Build the dual-head model ──────────────────────────────────────
    print("\nBuilding DualHeadECGFounder (routing='ptbxl_specific') ...")
    dual = DualHeadECGFounder(device, routing='ptbxl_specific')
    print(f"  {dual!r}")

    with open('tasks.txt') as f:
        heads = [l.strip() for l in f if l.strip()][:150]

    # ── 2. val_75deg comparison ──────────────────────────────────────────
    print("\n[1/2] val_75deg — does the dual model match each source on its own active heads?")
    z = np.load('data/fuzzylead2/ptbxl/val_75deg.npz')
    ecg = torch.from_numpy(z['ecg'].astype(np.float32))
    labels = z['labels']

    dl = DataLoader(TensorDataset(ecg), batch_size=64, shuffle=False)
    dual_p, base_p, fuzzy_p = [], [], []
    for (x,) in tqdm(dl, desc="  inference"):
        x = x.to(device)
        with torch.no_grad():
            dual_p.append (torch.sigmoid(dual      (x)).cpu().numpy())
            base_p.append (torch.sigmoid(dual.base (x)).cpu().numpy())
            fuzzy_p.append(torch.sigmoid(dual.fuzzy(x)).cpu().numpy())
    dual_p  = np.concatenate(dual_p)
    base_p  = np.concatenate(base_p)
    fuzzy_p = np.concatenate(fuzzy_p)

    fuzzy_idx = dual.fuzzy_head_indices
    active = np.where(labels.sum(axis=0) > 0)[0]
    print(f"\n  routed via fuzzy: {fuzzy_idx}")
    print(f"  routed via base : (all others, 142 heads)")
    print(f"\n  {'class':<46}{'pos':>4} {'src':>5}  {'BASE ROC':>9} {'FUZZY ROC':>10} {'DUAL ROC':>10}")
    print("  " + "-"*92)
    for idx in active:
        y = labels[:, idx]
        src = 'fuzzy' if idx in fuzzy_idx else 'base'
        b_roc = roc_auc_score(y, base_p[:, idx])
        f_roc = roc_auc_score(y, fuzzy_p[:, idx])
        d_roc = roc_auc_score(y, dual_p[:, idx])
        print(f"  {heads[idx][:46]:<46}{int(y.sum()):>4} {src:>5}  "
              f"{b_roc:>9.3f} {f_roc:>10.3f} {d_roc:>10.3f}"
              f"{'  ✓ matches '+src.upper() if abs(d_roc - (f_roc if src=='fuzzy' else b_roc)) < 1e-9 else ''}")

    # ── 3. fzark comparison ──────────────────────────────────────────────
    print("\n[2/2] fzark TP — V3.1 heads (non-fuzzy-routed should be bit-identical to BASE)")
    DATA_DIR = "data/ecg_tp_fzark"
    df = pd.read_csv(f"{DATA_DIR}/summary.csv")
    sampled = [g.sample(n=min(500, len(g)), random_state=42) for _, g in df.groupby('Event Type')]
    df = pd.concat(sampled).reset_index(drop=True)

    class TpDs(Dataset):
        def __init__(self):
            self.prep = ECGPreprocessor.for_fzark()
        def __len__(self): return len(df)
        def __getitem__(self, idx):
            rel = str(df.iloc[idx]['JSON File']).replace('\\','/')
            return self.prep.from_fzark_json(os.path.join(DATA_DIR, rel))

    dl = DataLoader(TpDs(), batch_size=64, shuffle=False, num_workers=0)
    dual_fzark, fuzzy_fzark = [], []
    for x in tqdm(dl, desc="  fzark inference"):
        x = x.to(device)
        with torch.no_grad():
            dual_fzark.append (torch.sigmoid(dual      (x)).cpu().numpy())
            fuzzy_fzark.append(torch.sigmoid(dual.fuzzy(x)).cpu().numpy())
    dual_fzark  = np.concatenate(dual_fzark)
    fuzzy_fzark = np.concatenate(fuzzy_fzark)
    base_fzark  = np.load('res/tp_fzark_full_suppression/baseline_probs_tp.npy')

    # Sanity: dual must equal base on ALL heads not routed to fuzzy
    non_fuzzy = [i for i in range(150) if i not in fuzzy_idx]
    max_diff_nonfuzzy = float(np.abs(dual_fzark[:, non_fuzzy] - base_fzark[:, non_fuzzy]).max())
    max_diff_fuzzy   = float(np.abs(dual_fzark[:, fuzzy_idx]  - fuzzy_fzark[:, fuzzy_idx]).max())
    print(f"\n  Max abs diff (DUAL vs BASE) on {len(non_fuzzy)} non-fuzzy heads:  {max_diff_nonfuzzy:.2e}  (should be 0)")
    print(f"  Max abs diff (DUAL vs FUZZY) on {len(fuzzy_idx)} fuzzy heads:     {max_diff_fuzzy:.2e}  (should be 0)")

    # V3.1 per-class detection rates
    print(f"\n  {'V3.1 Event':<35}{'idx':>4}{'n':>4}  {'src':>5}  {'BASE':>7} {'FUZZY':>7} {'DUAL':>7}")
    print("  " + "-"*82)
    for et, idx in sorted(FZARK_LABEL_MAP.items()):
        m = (df['Event Type']==et).values
        n = int(m.sum())
        if n == 0: continue
        src = 'fuzzy' if idx in fuzzy_idx else 'base'
        b = 100*(base_fzark[m, idx]   >= 0.5).mean()
        f = 100*(fuzzy_fzark[m, idx]  >= 0.5).mean()
        d = 100*(dual_fzark[m, idx]   >= 0.5).mean()
        print(f"  {et:<35}{idx:>4}{n:>4}  {src:>5}  {b:>6.1f}% {f:>6.1f}% {d:>6.1f}%")

    # Aggregate
    nm = sum((df['Event Type']==et).values.sum() for et,idx in FZARK_LABEL_MAP.items())
    for thr in [0.5, 0.6, 0.7]:
        b = sum(int((base_fzark[(df['Event Type']==et).values, idx]   >= thr).sum()) for et,idx in FZARK_LABEL_MAP.items())
        d = sum(int((dual_fzark[(df['Event Type']==et).values, idx]   >= thr).sum()) for et,idx in FZARK_LABEL_MAP.items())
        print(f"\n  AGGREGATE @t={thr}  BASE={100*b/nm:.1f}%   DUAL={100*d/nm:.1f}%  (Δ {100*(d-b)/nm:+.1f} pp)")

    # Persist
    os.makedirs('res/finetune_fuzzylead2', exist_ok=True)
    np.save('res/finetune_fuzzylead2/dual_probs_tp.npy', dual_fzark)
    print("\nSaved: res/finetune_fuzzylead2/dual_probs_tp.npy")


if __name__ == '__main__':
    main()
