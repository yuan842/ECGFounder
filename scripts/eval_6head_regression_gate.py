"""Regression gate for the 6-head v2 linear-probe checkpoint.

Three asymmetric tests (§5 of FINETUNE_STRATEGY_v2_6HEAD.md):

  1. 144 non-target heads must be bit-identical to base (Tier 1 guarantee).
  2. Per V3.1 event, candidate TP detection ≥ base − 0.5 pp.
  3. Per V3.1 event, candidate FP rate ≤ base + 0.5 pp.

A candidate is adopted only if all three pass.
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

TARGET_IDX  = [4, 5, 6, 9, 93, 98]
NON_TARGET  = [i for i in range(150) if i not in TARGET_IDX]
TOLERANCE   = 0.005   # 0.5 pp
import argparse
CKPT_DEFAULT = "checkpoint/1_lead_ECGFounder_6head_v2.pth"
TP_DIR      = "data/ecg_tp_fzark"
FP_DIR      = "data/ecg_fp_doctor removed1"


def sample_df(data_dir, per_class=500, seed=42):
    df = pd.read_csv(f"{data_dir}/summary.csv")
    sampled = [g.sample(n=min(per_class, len(g)), random_state=seed)
               for _, g in df.groupby('Event Type')]
    return pd.concat(sampled).reset_index(drop=True)


class FzarkDs(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.prep = ECGPreprocessor.for_fzark()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        rel = str(self.df.iloc[idx]['JSON File']).replace('\\','/')
        return self.prep.from_fzark_json(os.path.join(self.data_dir, rel))


def run_inference(ckpt, df, data_dir, device):
    model = load_ecgfounder(device, ckpt_path=ckpt); model.eval()
    dl = DataLoader(FzarkDs(df, data_dir), batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        for x in tqdm(dl, desc=f"  {os.path.basename(ckpt)[:30]}", leave=False):
            out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--out-suffix', default='',
                    help="Suffix on the output CSVs / npy filenames (e.g. '_v3_posw').")
    args = ap.parse_args()
    ckpt = args.ckpt
    suffix = args.out_suffix
    device = resolve_device()
    print(f"Device: {device}")
    print(f"Candidate checkpoint: {ckpt}")

    # ── Sample (cached probs are 500/class @ seed 42 → matches this sampling)
    print("\nSampling fzark cohorts ...")
    tp_df = sample_df(TP_DIR)
    fp_df = sample_df(FP_DIR)
    print(f"  TP: {len(tp_df)} records, FP: {len(fp_df)} records")

    # ── Cached BASE predictions
    base_tp = np.load("res/tp_fzark_full_suppression/baseline_probs_tp.npy")
    base_fp = np.load("res/fp_allclass_full_suppression/baseline_probs_full.npy")
    assert base_tp.shape[0] == len(tp_df), (base_tp.shape, len(tp_df))
    assert base_fp.shape[0] == len(fp_df), (base_fp.shape, len(fp_df))

    # ── Candidate predictions
    print("\nRunning candidate on TP cohort ...")
    cand_tp = run_inference(ckpt, tp_df, TP_DIR, device)
    print("Running candidate on FP cohort ...")
    cand_fp = run_inference(ckpt, fp_df, FP_DIR, device)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1 — non-target heads bit-identical
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST 1: non-target heads must be bit-identical to base")
    print("="*80)
    tp_max_diff_nt = float(np.abs(cand_tp[:, NON_TARGET] - base_tp[:, NON_TARGET]).max())
    fp_max_diff_nt = float(np.abs(cand_fp[:, NON_TARGET] - base_fp[:, NON_TARGET]).max())
    print(f"  TP cohort, max abs diff across {len(NON_TARGET)} non-target heads: {tp_max_diff_nt:.2e}")
    print(f"  FP cohort, max abs diff across {len(NON_TARGET)} non-target heads: {fp_max_diff_nt:.2e}")
    test1_pass = (tp_max_diff_nt < 1e-6) and (fp_max_diff_nt < 1e-6)
    print(f"  TEST 1: {'✅ PASS' if test1_pass else '❌ FAIL'}")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2 — Per V3.1 event TP detection ≥ base − 0.5 pp
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST 2: per-V3.1-event TP detection must not regress > 0.5 pp")
    print("="*80)
    print(f"  {'Event':<35}{'idx':>4}{'n':>5}  {'BASE':>7} {'CAND':>7} {'Δ':>7}  status")
    rows_tp = []
    test2_pass = True
    for et, idx in sorted(FZARK_LABEL_MAP.items()):
        m = (tp_df['Event Type'] == et).values
        n = int(m.sum())
        if n == 0: continue
        b = float((base_tp[m, idx] >= 0.5).mean())
        c = float((cand_tp[m, idx] >= 0.5).mean())
        delta = c - b
        status = "✓" if delta >= -TOLERANCE else "✗ REGRESSION"
        if delta < -TOLERANCE: test2_pass = False
        is_target = idx in TARGET_IDX
        marker = "*" if is_target else " "
        print(f"  {marker}{et:<34}{idx:>4}{n:>5}  {100*b:>6.1f}% {100*c:>6.1f}% {100*delta:+6.1f}pp  {status}")
        rows_tp.append(dict(event_type=et, idx=idx, is_target=is_target,
                             n=n, base_det=b, cand_det=c, delta_pp=100*delta))
    print(f"  TEST 2: {'✅ PASS' if test2_pass else '❌ FAIL'}  (* = target head)")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3 — Per V3.1 event FP rate ≤ base + 0.5 pp
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST 3: per-V3.1-event FP rate must not increase > 0.5 pp")
    print("="*80)
    print(f"  {'Event':<35}{'idx':>4}{'n':>5}  {'BASE':>7} {'CAND':>7} {'Δ':>7}  status")
    rows_fp = []
    test3_pass = True
    for et, idx in sorted(FZARK_LABEL_MAP.items()):
        m = (fp_df['Event Type'] == et).values
        n = int(m.sum())
        if n == 0: continue
        b = float((base_fp[m, idx] >= 0.5).mean())
        c = float((cand_fp[m, idx] >= 0.5).mean())
        delta = c - b
        status = "✓" if delta <= TOLERANCE else "✗ REGRESSION"
        if delta > TOLERANCE: test3_pass = False
        is_target = idx in TARGET_IDX
        marker = "*" if is_target else " "
        print(f"  {marker}{et:<34}{idx:>4}{n:>5}  {100*b:>6.1f}% {100*c:>6.1f}% {100*delta:+6.1f}pp  {status}")
        rows_fp.append(dict(event_type=et, idx=idx, is_target=is_target,
                             n=n, base_fp_rate=b, cand_fp_rate=c, delta_pp=100*delta))
    print(f"  TEST 3: {'✅ PASS' if test3_pass else '❌ FAIL'}")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL VERDICT + persist
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"FINAL VERDICT: {'✅ ADOPT' if (test1_pass and test2_pass and test3_pass) else '❌ REJECT'}")
    print("="*80)
    print(f"  Test 1 (non-target bit-identical):  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (TP detection preserved):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (FP rate preserved):         {'PASS' if test3_pass else 'FAIL'}")

    # Save reports
    out_dir = "res/finetune_6head_v2"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows_tp).to_csv(f"{out_dir}/regression_gate_TP{suffix}.csv", index=False)
    pd.DataFrame(rows_fp).to_csv(f"{out_dir}/regression_gate_FP{suffix}.csv", index=False)
    np.save(f"{out_dir}/cand_probs_tp{suffix}.npy", cand_tp)
    np.save(f"{out_dir}/cand_probs_fp{suffix}.npy", cand_fp)
    print(f"\nCSVs and probability matrices written under {out_dir}/")


if __name__ == '__main__':
    main()
