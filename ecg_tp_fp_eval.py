#!/usr/bin/env python3
"""
ECG TP/FP Dataset Evaluation Script
=====================================
Evaluates 3 ECGFounder model variants on two labeled ECG datasets:
  - ecg-tp  : ECGs confirmed as TRUE  POSITIVES by clinicians (ground-truth label = 1)
  - ecg-fp  : ECGs confirmed as FALSE POSITIVES by clinicians (ground-truth label = 0)

Metrics are computed at two fixed thresholds (0.5 and 0.8) for every one of
the 150 ECGFounder tasks, plus a summary table averaged across tasks.

Usage (example)
---------------
python ecg_tp_fp_eval.py \
    --tp_dir   "data/ecg-tp_rex" \
    --fp_dir   "data/ecg_fp_doctor removed1" \
    --ckpt1    checkpoint/12_lead_ECGFounder.pth \
    --ckpt2    checkpoint/1_lead_ECGFounder.pth \
    --ckpt3    checkpoint/finetuned_ECGFounder.pth \
    --task_idx -1 \
    --output   res/tp_fp_eval

Arguments
---------
--tp_dir   : folder containing WFDB ECG files for true-positive ECGs
--fp_dir   : folder containing WFDB ECG files for false-positive ECGs
--ckpt1    : checkpoint for Model 1 (12-lead ECGFounder, default)
--ckpt2    : checkpoint for Model 2 (1-lead ECGFounder, default)
--ckpt3    : checkpoint for Model 3 (fine-tuned or 3rd variant)
--task_idx : which of the 150 tasks to report in detail (-1 = all tasks)
--output   : directory for result CSVs and summary
--batch_size: inference batch size (default 64)
--fs_target : target sampling frequency the model was trained on (default 5000)
"""

import os
import sys
import glob
import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import medfilt, iirnotch, filtfilt, butter
from scipy.interpolate import interp1d
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, f1_score
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Import project modules
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from net1d import Net1D


# ─────────────────────────────────────────────────────────────────────────────
# Signal preprocessing (mirrors dataset.py / util.py)
# ─────────────────────────────────────────────────────────────────────────────

def filter_bandpass(ecg, fs):
    """Multi-stage ECG filter: notch (50 Hz) → bandpass (0.67–40 Hz) → baseline."""
    b_notch, a_notch = iirnotch(50, 30, fs)
    filtered = np.zeros_like(ecg)
    for c in range(ecg.shape[0]):
        filtered[c] = filtfilt(b_notch, a_notch, ecg[c])

    b_bp, a_bp = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(ecg.shape[0]):
        filtered[c] = filtfilt(b_bp, a_bp, filtered[c])

    baseline = np.zeros_like(filtered)
    for c in range(filtered.shape[0]):
        k = int(0.4 * fs) + 1
        if k % 2 == 0:
            k += 1
        baseline[c] = medfilt(filtered[c], kernel_size=k)
    return filtered - baseline


def z_score(ecg):
    return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)


def resample_ecg(ecg, fs_in, fs_out):
    """Resample ECG (channels × samples) from fs_in to fs_out."""
    if fs_in == fs_out or fs_in == 0 or ecg.shape[1] == 0:
        return ecg
    fs_in, fs_out = int(fs_in), int(fs_out)
    if 2 * fs_out == fs_in:                  # simple 2× downsampling
        return ecg[:, ::2]
    t = ecg.shape[1] / fs_in
    x_old = np.linspace(0, t, ecg.shape[1], endpoint=True)
    x_new = np.linspace(0, t, int(t * fs_out), endpoint=True)
    out = np.zeros((ecg.shape[0], len(x_new)))
    for c in range(ecg.shape[0]):
        out[c] = interp1d(x_old, ecg[c], kind='linear', fill_value='extrapolate')(x_new)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_LEADS_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Alternative lead name variants seen in different datasets
LEAD_ALIASES = {
    'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF',
    'avr': 'aVR', 'avl': 'aVL', 'avf': 'aVF',
    'i': 'I', 'ii': 'II', 'iii': 'III',
}


def normalise_lead_name(name):
    return LEAD_ALIASES.get(name, name)


class ECGFolderDataset(Dataset):
    """
    Scans a directory (recursively) for WFDB .hea files and loads each ECG.

    Parameters
    ----------
    folder   : path to the directory
    label    : binary label applied to EVERY file in the folder (1 or 0)
    n_leads  : 12 (12-lead model) or 1 (Lead-I model)
    fs_target: target sampling frequency (Hz); default 5000
    """

    def __init__(self, folder: str, label: int, n_leads: int = 12,
                 fs_target: int = 5000):
        self.label = label
        self.n_leads = n_leads
        self.fs_target = fs_target

        # Collect all header files
        hea_files = sorted(glob.glob(os.path.join(folder, '**', '*.hea'),
                                     recursive=True))
        # Strip the .hea extension to get the record name
        self.records = [f[:-4] for f in hea_files]

        if len(self.records) == 0:
            raise FileNotFoundError(
                f"No WFDB .hea files found in: {folder}\n"
                f"Please confirm the path is correct and contains WFDB records."
            )
        print(f"  Found {len(self.records)} records in '{os.path.basename(folder)}'")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        try:
            signal, meta = wfdb.rdsamp(record_path)
        except Exception as e:
            # Return zeros on read error so the batch still works
            dummy = np.zeros((self.n_leads, self.fs_target), dtype=np.float32)
            return torch.FloatTensor(dummy), torch.tensor(self.label, dtype=torch.float32)

        fs_in = meta['fs']
        sig_names = [normalise_lead_name(s.strip()) for s in meta['sig_name']]

        # shape: (samples, channels) → (channels, samples)
        ecg = np.nan_to_num(signal.T, nan=0.0)

        # ── lead selection ──────────────────────────────────────────────────
        if self.n_leads == 12:
            target_leads = STANDARD_LEADS_12
        else:
            target_leads = ['I']

        indices = []
        for lead in target_leads:
            if lead in sig_names:
                indices.append(sig_names.index(lead))
            else:
                indices.append(0)   # fall back to first channel if missing

        ecg = ecg[indices, :]       # (n_leads, samples)

        # ── preprocessing ───────────────────────────────────────────────────
        ecg = filter_bandpass(ecg, fs_in)
        ecg = resample_ecg(ecg, fs_in, self.fs_target)
        # Trim or pad to exactly fs_target samples (10 s at 500 Hz = 5000)
        target_len = self.fs_target
        if ecg.shape[1] > target_len:
            ecg = ecg[:, :target_len]
        elif ecg.shape[1] < target_len:
            pad = np.zeros((ecg.shape[0], target_len - ecg.shape[1]))
            ecg = np.concatenate([ecg, pad], axis=1)

        ecg = z_score(ecg).astype(np.float32)
        return (torch.FloatTensor(ecg),
                torch.tensor(float(self.label), dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_ecgfounder(n_leads: int, n_classes: int = 150) -> Net1D:
    return Net1D(
        in_channels=n_leads,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes,
    )


MODEL_CONFIGS = {
    # name            : (n_leads, description)
    'ECGFounder-12L'  : (12, '12-lead ECGFounder'),
    'ECGFounder-1L'   : (1,  '1-lead ECGFounder'),
    'ECGFounder-FT'   : (12, 'Fine-tuned / 3rd-variant ECGFounder'),
}


def load_model(ckpt_path: str, n_leads: int, device):
    model = build_ecgfounder(n_leads)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """Returns (all_probs, all_labels) as numpy arrays."""
    probs_list, labels_list = [], []
    with torch.no_grad():
        for ecg, label in tqdm(loader, desc='  Inference', leave=False):
            ecg = ecg.to(device)
            logits = model(ecg)                     # (B, 150)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(prob)
            labels_list.append(label.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_binary(gt_1d, prob_1d, threshold):
    """
    Compute classification metrics for a single task.

    Parameters
    ----------
    gt_1d   : 1-D array of {0, 1} ground-truth labels
    prob_1d : 1-D array of predicted probabilities
    threshold: float

    Returns dict with: sensitivity, specificity, ppv, npv, f1, auroc, auprc,
                       tp, fp, tn, fn, n_pos, n_neg
    """
    pred = (prob_1d >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(gt_1d, pred, labels=[0, 1]).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
    elif gt_1d.sum() == 0:          # no positives in ground truth
        tn, fp, fn, tp = len(gt_1d) - pred.sum(), pred.sum(), 0, 0
    else:                           # no negatives in ground truth
        tn, fp, fn, tp = 0, 0, len(gt_1d) - pred.sum(), pred.sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv         = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    f1          = (2 * ppv * sensitivity / (ppv + sensitivity)
                   if (ppv + sensitivity) > 0 else float('nan'))

    try:
        auroc = roc_auc_score(gt_1d, prob_1d)
    except ValueError:
        auroc = float('nan')

    try:
        auprc = average_precision_score(gt_1d, prob_1d)
    except ValueError:
        auprc = float('nan')

    return dict(
        sensitivity=sensitivity, specificity=specificity,
        ppv=ppv, npv=npv, f1=f1,
        auroc=auroc, auprc=auprc,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        n_pos=int(gt_1d.sum()), n_neg=int((1 - gt_1d).sum()),
    )


def evaluate_all_tasks(gt, probs, tasks, thresholds=(0.5, 0.8)):
    """
    gt     : (N,) binary array  (all 1s from TP set + all 0s from FP set)
    probs  : (N, 150) probability array
    tasks  : list of 150 task names
    Returns dict[threshold] → DataFrame (one row per task)
    """
    results = {}
    for thr in thresholds:
        rows = []
        for i, task in enumerate(tasks):
            m = compute_metrics_binary(gt, probs[:, i], thr)
            m['task'] = task
            m['task_idx'] = i
            m['threshold'] = thr
            rows.append(m)
        results[thr] = pd.DataFrame(rows)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ['sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'auroc', 'auprc']


def print_summary_table(model_results: dict, thresholds=(0.5, 0.8)):
    """
    model_results: {model_name: {threshold: DataFrame}}
    Prints a compact summary table averaged across the 150 tasks.
    """
    header = f"\n{'─'*80}"
    print(header)
    print(f"{'Model':<22} {'Threshold':>10}  "
          + "  ".join(f"{m:>10}" for m in METRIC_COLS))
    print(f"{'─'*80}")
    for mname, thr_dict in model_results.items():
        for thr in thresholds:
            df = thr_dict[thr]
            means = df[METRIC_COLS].mean()
            row = f"{mname:<22} {thr:>10.1f}  " + \
                  "  ".join(f"{means[m]:>10.4f}" for m in METRIC_COLS)
            print(row)
        print()
    print(f"{'─'*80}\n")


def print_per_dataset_summary(gt, probs, tasks, model_name, thresholds=(0.5, 0.8)):
    """
    Also break down TP-set vs FP-set separately.
    gt_full contains n_tp ones followed by n_fp zeros.
    We split and report sensitivity (on TP) and specificity (on FP) separately.
    """
    pass   # done inline in main()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ECG TP/FP evaluation')

    p.add_argument('--tp_dir',   required=True,
                   help='Directory containing WFDB ECG files for true-positive cases')
    p.add_argument('--fp_dir',   required=True,
                   help='Directory containing WFDB ECG files for false-positive cases')

    p.add_argument('--ckpt1', required=True,
                   help='Checkpoint path for Model 1 (12-lead ECGFounder)')
    p.add_argument('--ckpt2', required=True,
                   help='Checkpoint path for Model 2 (1-lead ECGFounder)')
    p.add_argument('--ckpt3', required=True,
                   help='Checkpoint path for Model 3 (fine-tuned / 3rd variant)')

    p.add_argument('--model3_leads', type=int, default=12,
                   help='Number of leads for Model 3 (default: 12)')

    p.add_argument('--task_idx', type=int, default=-1,
                   help='Task index to highlight in detailed output (-1 = all)')
    p.add_argument('--tasks_file', default='tasks.txt',
                   help='Path to tasks.txt (default: tasks.txt)')

    p.add_argument('--output', default='res/tp_fp_eval',
                   help='Output directory for CSVs')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--fs_target', type=int, default=5000,
                   help='Target sampling frequency in Hz (default: 5000 = 500 Hz × 10 s)')
    p.add_argument('--gpu', type=int, default=0,
                   help='GPU index (-1 for CPU)')

    return p.parse_args()


def main():
    args = parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # ── output directory ────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── tasks ───────────────────────────────────────────────────────────────
    tasks_path = args.tasks_file
    if not os.path.isabs(tasks_path):
        tasks_path = os.path.join(os.path.dirname(__file__), tasks_path)
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            t = line.strip()
            if t:
                tasks.append(t)
    n_tasks = len(tasks)
    print(f"Tasks loaded: {n_tasks}")

    # ── model registry ──────────────────────────────────────────────────────
    model_registry = {
        'ECGFounder-12L': (args.ckpt1, 12),
        'ECGFounder-1L' : (args.ckpt2, 1),
        'ECGFounder-FT' : (args.ckpt3, args.model3_leads),
    }

    thresholds = [0.5, 0.8]

    # ── collect all results ─────────────────────────────────────────────────
    all_model_results = {}          # {model_name: {thresh: df}}
    all_model_probs   = {}          # {model_name: (probs_tp, probs_fp)}
    combined_gt       = None        # set once, reused

    for model_name, (ckpt_path, n_leads) in model_registry.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({n_leads}-lead)")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # ── load model ──────────────────────────────────────────────────────
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue
        model = load_model(ckpt_path, n_leads, device)

        # ── datasets ────────────────────────────────────────────────────────
        print("\nLoading TP dataset …")
        ds_tp = ECGFolderDataset(args.tp_dir, label=1,
                                 n_leads=n_leads, fs_target=args.fs_target)
        print("Loading FP dataset …")
        ds_fp = ECGFolderDataset(args.fp_dir, label=0,
                                 n_leads=n_leads, fs_target=args.fs_target)

        loader_tp = DataLoader(ds_tp, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
        loader_fp = DataLoader(ds_fp, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        # ── inference ───────────────────────────────────────────────────────
        print("\nRunning inference on TP set …")
        probs_tp, labels_tp = run_inference(model, loader_tp, device)

        print("Running inference on FP set …")
        probs_fp, labels_fp = run_inference(model, loader_fp, device)

        # Combined arrays (TP rows first, FP rows second)
        probs_all  = np.concatenate([probs_tp, probs_fp], axis=0)   # (N, 150)
        labels_all = np.concatenate([labels_tp, labels_fp], axis=0) # (N,)

        if combined_gt is None:
            combined_gt = labels_all   # same for every model

        all_model_probs[model_name] = (probs_tp, probs_fp)

        # ── per-task metrics at both thresholds ─────────────────────────────
        task_results = evaluate_all_tasks(labels_all, probs_all,
                                          tasks, thresholds)
        all_model_results[model_name] = task_results

        # ── save per-model CSVs ─────────────────────────────────────────────
        for thr in thresholds:
            out_csv = os.path.join(
                args.output,
                f"{model_name}_thr{thr}_all_tasks.csv".replace('-', '_')
            )
            task_results[thr].to_csv(out_csv, index=False, float_format='%.4f')
            print(f"  Saved: {out_csv}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_model_results:
        print("\n[ERROR] No models could be evaluated (checkpoints missing).")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Summary table: mean across 150 tasks
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY — Mean metrics across all 150 tasks")
    print("="*80)
    print_summary_table(all_model_results, thresholds)

    # ─────────────────────────────────────────────────────────────────────────
    # TP-set sensitivity vs FP-set specificity breakdown
    # ─────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("TP-set Sensitivity  vs  FP-set Specificity  (mean across tasks)")
    print("="*80)
    print(f"{'Model':<22} {'Threshold':>10}  {'Sensitivity(TP)':>17}  {'Specificity(FP)':>17}")
    print(f"{'─'*80}")

    sep_rows = []
    for model_name, (probs_tp, probs_fp) in all_model_probs.items():
        n_tp = probs_tp.shape[0]
        n_fp = probs_fp.shape[0]
        gt_tp = np.ones(n_tp)
        gt_fp = np.zeros(n_fp)

        for thr in thresholds:
            sens_per_task = []
            spec_per_task = []
            for i in range(n_tasks):
                pred_tp = (probs_tp[:, i] >= thr).astype(int)
                tp = pred_tp.sum()
                fn = n_tp - tp
                sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                sens_per_task.append(sens)

                pred_fp = (probs_fp[:, i] >= thr).astype(int)
                fp_cnt = pred_fp.sum()
                tn = n_fp - fp_cnt
                spec = tn / (tn + fp_cnt) if (tn + fp_cnt) > 0 else float('nan')
                spec_per_task.append(spec)

            mean_sens = np.nanmean(sens_per_task)
            mean_spec = np.nanmean(spec_per_task)
            print(f"{model_name:<22} {thr:>10.1f}  {mean_sens:>17.4f}  {mean_spec:>17.4f}")
            sep_rows.append(dict(
                model=model_name, threshold=thr,
                mean_sensitivity_on_tp=round(mean_sens, 4),
                mean_specificity_on_fp=round(mean_spec, 4),
            ))
        print()

    sep_df = pd.DataFrame(sep_rows)
    sep_csv = os.path.join(args.output, 'tp_fp_sensitivity_specificity.csv')
    sep_df.to_csv(sep_csv, index=False, float_format='%.4f')
    print(f"Saved: {sep_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # Highlighted task (if requested)
    # ─────────────────────────────────────────────────────────────────────────
    if args.task_idx >= 0 and args.task_idx < n_tasks:
        print(f"\n{'='*80}")
        print(f"Detail for Task {args.task_idx}: {tasks[args.task_idx]}")
        print(f"{'='*80}")
        print(f"{'Model':<22} {'Threshold':>10}  "
              + "  ".join(f"{m:>10}" for m in METRIC_COLS)
              + "  " + "  ".join(f"{m:>5}" for m in ['TP', 'FP', 'TN', 'FN']))
        print("─"*80)
        for model_name, thr_dict in all_model_results.items():
            for thr in thresholds:
                row = thr_dict[thr][thr_dict[thr].task_idx == args.task_idx].iloc[0]
                print(
                    f"{model_name:<22} {thr:>10.1f}  "
                    + "  ".join(f"{row[m]:>10.4f}" for m in METRIC_COLS)
                    + "  " + f"{row.tp:>5} {row.fp:>5} {row.tn:>5} {row.fn:>5}"
                )
            print()

    # ─────────────────────────────────────────────────────────────────────────
    # Master comparison CSV (all models × all thresholds × all tasks)
    # ─────────────────────────────────────────────────────────────────────────
    master_rows = []
    for model_name, thr_dict in all_model_results.items():
        for thr in thresholds:
            df = thr_dict[thr].copy()
            df.insert(0, 'model', model_name)
            master_rows.append(df)

    master_df = pd.concat(master_rows, ignore_index=True)
    master_csv = os.path.join(args.output, 'master_results.csv')
    master_df.to_csv(master_csv, index=False, float_format='%.4f')
    print(f"\nMaster results saved: {master_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary CSV (mean across tasks)
    # ─────────────────────────────────────────────────────────────────────────
    summary_rows = []
    for model_name, thr_dict in all_model_results.items():
        for thr in thresholds:
            means = thr_dict[thr][METRIC_COLS].mean()
            entry = {'model': model_name, 'threshold': thr}
            entry.update({m: round(means[m], 4) for m in METRIC_COLS})
            entry['n_tp_samples'] = int((combined_gt == 1).sum())
            entry['n_fp_samples'] = int((combined_gt == 0).sum())
            summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output, 'summary_results.csv')
    summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"Summary results saved: {summary_csv}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()
