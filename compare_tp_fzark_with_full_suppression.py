"""
compare_tp_fzark_with_full_suppression.py
==========================================

Run the baseline threshold variant comparison on `ecg_tp_fzark`
(clinician-confirmed true positives).

Same pipeline as compare_all_classes_with_full_suppression.py but on
TP data where **higher retention is better**.

For each threshold (0.5, 0.6, 0.7):
  - An event is a "detected TP" if p >= threshold  (want: high)
  - A detected TP that the suppressor keeps is "retained"  (want: high)
  - A detected TP that the suppressor suppresses is a "TP loss"  (want: low)
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from multiclass_fp_suppression import MultiClassFPSuppressor, ACTIVE_RULES
from device_utils import resolve_device
from preprocessing import ECGPreprocessor, TARGET_LEN
from checkpoints import load_ecgfounder
from label_config import FZARK_LABEL_MAP as LABEL_MAP

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR        = "./data/ecg_tp_fzark"
SUMMARY_CSV     = "./data/ecg_tp_fzark/summary.csv"
BASELINE_CKPT   = "./checkpoint/1_lead_ECGFounder.pth"
OUT_DIR         = "./res/tp_fzark_full_suppression"

AFIB_IDX = 5

MODEL_THRESHOLDS = {
    "baseline":   0.5,
}

# Baseline threshold variants for sensitivity analysis
BASELINE_THRESHOLD_VARIANTS = [0.5, 0.6, 0.7]

# Per-class suppression — which event types have an active v2 filter rule
SUPPRESSED_EVENTS = set(ACTIVE_RULES.keys())  # AFib, Bradycardia, SV Trigeminy, V Trigeminy

BATCH_SIZE     = 64


class TpDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.prep = ECGPreprocessor.for_fzark()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row['JSON File']).replace('\\', '/')
        path = os.path.join(self.data_dir, rel)
        try:
            tensor = self.prep.from_fzark_json(path)
        except Exception:
            tensor = torch.zeros(1, TARGET_LEN)
        return tensor, idx


@torch.no_grad()
def run_inference(model, loader, device):
    out = []
    for x, _ in tqdm(loader, desc="  inference"):
        logits = model(x.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200,
                    help='Max samples per event type (0 = all)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--suppression', choices=['on', 'off'], default='on',
                    help='Enable the v2 feature-gate FP suppression layer. '
                         'When off, all alerts pass through unchanged — useful '
                         'for A/B comparison with the suppression layer disabled.')
    args = ap.parse_args()
    suppression_enabled = (args.suppression == 'on')

    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}\n")

    # ── Load and optionally stratified-sample TP dataset ───────────────────
    df = pd.read_csv(SUMMARY_CSV)
    if args.per_class > 0:
        sampled = [g.sample(n=min(args.per_class, len(g)), random_state=args.seed)
                   for _, g in df.groupby('Event Type')]
        df = pd.concat(sampled).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    print(f"TP cohort: {len(df)} records across {df['Event Type'].nunique()} event types")
    print(df['Event Type'].value_counts().to_string()); print()

    ds = TpDataset(df, DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Run baseline model ──────────────────────────────────────────────────
    probs_dict = {}
    for name, ckpt in [('baseline', BASELINE_CKPT)]:
        print(f"\n{'='*72}\nMODEL: {name.upper()}\n{'='*72}")
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} not found"); continue
        model = load_ecgfounder(device, ckpt_path=ckpt)
        probs = run_inference(model, dl, device)
        np.save(os.path.join(OUT_DIR, f"{name}_probs_tp.npy"), probs)
        probs_dict[name] = probs

    if 'baseline' not in probs_dict:
        print("Need baseline model — exiting."); return

    # ── Suppressor (multiclass router) ──────────────────────────────────────
    suppressor = MultiClassFPSuppressor(enabled=suppression_enabled)
    # Local view of "which event types are currently being gated". Empty when
    # suppression is off — drives the has_suppressor branches below.
    active_events = set(suppressor.supported_classes())
    print(f"\nSuppression: {'ON' if suppression_enabled else 'OFF'}")
    print(f"Suppressor supports: {sorted(active_events)}\n")

    # Pre-compute suppressor decisions for ALL baseline alerts at the lowest
    # threshold (0.5).  Higher thresholds are subsets.
    baseline_probs = probs_dict['baseline']
    min_thr = min(BASELINE_THRESHOLD_VARIANTS)

    print("Pre-computing suppressor decisions for baseline model alerts...")
    decisions = {}      # event_idx → SuppressionResult
    for i, row in tqdm(df.iterrows(), total=len(df), desc="  suppressor"):
        et = row['Event Type']
        rel = str(row['JSON File']).replace('\\','/')
        path = os.path.join(DATA_DIR, rel)
        # v2: route by event type name directly
        if et not in active_events:
            continue   # no rule for this class
        cls_idx = LABEL_MAP.get(et, AFIB_IDX)
        p = float(baseline_probs[i, cls_idx])
        if p < min_thr:
            continue   # not an alert at any threshold
        try:
            r = suppressor.suppress_alert(et, p, path)
            decisions[int(i)] = r
        except Exception:
            decisions[int(i)] = None

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: per-class TP retention at default threshold
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*108)
    print("TP RETENTION  (baseline t=0.5)")
    print("  Higher detection & retention = better  (these are confirmed TPs)")
    print("="*108)

    default_rows = []
    for et, grp in df.groupby('Event Type'):
        idxs = grp.index.values
        n_total = len(idxs)
        in_map = et in LABEL_MAP
        cls_idx = LABEL_MAP.get(et, AFIB_IDX)

        # v2: route by event type name directly
        has_suppressor = et in active_events

        row = dict(event_type=et, n=n_total, in_label_map=in_map,
                   class_idx=cls_idx,
                   suppressor_target=et if has_suppressor else 'none')

        # Baseline at default threshold
        p_tgt_bs = baseline_probs[idxs, cls_idx]
        bs_thr = MODEL_THRESHOLDS['baseline']
        detected_bs = int((p_tgt_bs >= bs_thr).sum())
        row['baseline_detected'] = detected_bs
        row['baseline_detect_pct'] = 100 * detected_bs / max(1, n_total)
        if has_suppressor:
            retained_bs = 0
            for pos, ei in enumerate(idxs):
                if p_tgt_bs[pos] < bs_thr: continue
                sr = decisions.get(int(ei))
                if sr is None or sr.keep: retained_bs += 1
            row['baseline_retained'] = retained_bs
            row['baseline_retain_pct'] = 100 * retained_bs / max(1, n_total)
            row['baseline_supp_loss_pct'] = (100 * (detected_bs - retained_bs) /
                                              max(1, detected_bs)) if detected_bs > 0 else 0.0
        else:
            row['baseline_retained'] = detected_bs
            row['baseline_retain_pct'] = 100 * detected_bs / max(1, n_total)
            row['baseline_supp_loss_pct'] = 0.0

        default_rows.append(row)

    default_df = pd.DataFrame(default_rows)
    default_df = default_df.sort_values(['in_label_map', 'event_type'], ascending=[False, True])

    # Console
    print(f"\n{'Event Type':<33} {'n':>5} {'route':>16}  "
          f"{'det%':>7} {'ret%':>7} {'loss%':>8}")
    print("-"*90)
    for _, r in default_df.iterrows():
        marker = "*" if r['in_label_map'] else " "
        route = (r['suppressor_target'][:14]
                 if r['suppressor_target'] != 'none' else 'no_supp')
        print(f"{r['event_type']:<33} {r['n']:>5} {route:>16}  "
              f"{r['baseline_detect_pct']:>7.1f} {r['baseline_retain_pct']:>7.1f} "
              f"{r['baseline_supp_loss_pct']:>8.1f}  {marker}")
    print("-"*90)

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE THRESHOLD VARIANT COMPARISON ON TPs
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "="*120)
    print("BASELINE THRESHOLD VARIANT COMPARISON  on  ecg_tp_fzark  (TP retention)")
    print(f"  Thresholds: {BASELINE_THRESHOLD_VARIANTS}")
    print("  Higher detection & retention = better")
    print("="*120)

    variant_rows = []
    for et, grp in df.groupby('Event Type'):
        idxs = grp.index.values
        n_total = len(idxs)
        in_map = et in LABEL_MAP
        cls_idx = LABEL_MAP.get(et, AFIB_IDX)

        # v2: route by event type name directly
        has_suppressor = et in active_events

        vrow = dict(event_type=et, n=n_total, in_label_map=in_map,
                    class_idx=cls_idx,
                    suppressor_target=et if has_suppressor else 'none')

        p_tgt = baseline_probs[idxs, cls_idx]

        for thr in BASELINE_THRESHOLD_VARIANTS:
            tag = f"bs_{thr:.1f}"
            detected = int((p_tgt >= thr).sum())
            vrow[f'{tag}_detected']    = detected
            vrow[f'{tag}_detect_pct']  = 100 * detected / max(1, n_total)

            if has_suppressor:
                retained = 0
                for pos, ei in enumerate(idxs):
                    if p_tgt[pos] < thr: continue
                    sr = decisions.get(int(ei))
                    if sr is None or sr.keep: retained += 1
                vrow[f'{tag}_retained']   = retained
                vrow[f'{tag}_retain_pct'] = 100 * retained / max(1, n_total)
                vrow[f'{tag}_supp_loss_pct'] = (100 * (detected - retained) /
                                                 max(1, detected)) if detected > 0 else 0.0
            else:
                vrow[f'{tag}_retained']   = detected
                vrow[f'{tag}_retain_pct'] = 100 * detected / max(1, n_total)
                vrow[f'{tag}_supp_loss_pct'] = 0.0
        variant_rows.append(vrow)

    var_df = pd.DataFrame(variant_rows)
    var_df = var_df.sort_values(['in_label_map', 'event_type'], ascending=[False, True])
    out_suffix = "" if suppression_enabled else "_supp_off"
    var_csv = os.path.join(OUT_DIR, f'tp_baseline_threshold_variants{out_suffix}.csv')
    var_df.to_csv(var_csv, index=False)

    # Console table
    thr_tags = [f"bs_{t:.1f}" for t in BASELINE_THRESHOLD_VARIANTS]
    thr_hdr  = "  ".join(f"  t={t:.1f}              " for t in BASELINE_THRESHOLD_VARIANTS)
    col_hdr  = "  ".join(f"{'det%':>6} {'ret%':>6} {'loss%':>6}" for _ in BASELINE_THRESHOLD_VARIANTS)
    print(f"\n{'Event Type':<33} {'n':>5}   {thr_hdr}")
    print(f"{'':33} {'':>5}   {col_hdr}")
    print("-"*130)
    for _, r in var_df.iterrows():
        parts = []
        for tag in thr_tags:
            det_p  = r[f'{tag}_detect_pct']
            ret_p  = r[f'{tag}_retain_pct']
            loss_p = r[f'{tag}_supp_loss_pct']
            parts.append(f"{det_p:>6.1f} {ret_p:>6.1f} {loss_p:>6.1f}")
        marker = "*" if r['in_label_map'] else " "
        print(f"{r['event_type']:<33} {r['n']:>5}   {'  '.join(parts)} {marker}")
    print("-"*130)
    print("det% = model detection rate  |  ret% = retained after suppression  |  loss% = suppressor TP loss")
    print("* = native-class head")

    # Aggregate
    print("\nAggregate TP retention (all classes, weighted by n):")
    tot_n = var_df['n'].sum()
    print(f"  {'threshold':>10}  {'detection':>20}  {'retention (post-supp)':>26}  {'supp TP loss':>16}")
    print(f"  {'─'*10}  {'─'*20}  {'─'*26}  {'─'*16}")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        det  = var_df[f'{tag}_detected'].sum()
        ret  = var_df[f'{tag}_retained'].sum()
        loss = det - ret
        print(f"  {thr:>10.1f}  {100*det/tot_n:>6.1f}% ({int(det):>5}/{tot_n})  "
              f"{100*ret/tot_n:>6.1f}% ({int(ret):>5}/{tot_n})  "
              f"{100*loss/max(1,det):>6.1f}% ({int(loss):>4}/{int(det)})")

    # Suppressed classes only
    supp_v = var_df[var_df['suppressor_target'] != 'none']
    supp_n = supp_v['n'].sum()
    print(f"\nSuppressed classes only (n={supp_n}):")
    print(f"  {'threshold':>10}  {'detection':>20}  {'retention (post-supp)':>26}  {'supp TP loss':>16}")
    print(f"  {'─'*10}  {'─'*20}  {'─'*26}  {'─'*16}")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        det  = supp_v[f'{tag}_detected'].sum()
        ret  = supp_v[f'{tag}_retained'].sum()
        loss = det - ret
        print(f"  {thr:>10.1f}  {100*det/supp_n:>6.1f}% ({int(det):>5}/{supp_n})  "
              f"{100*ret/supp_n:>6.1f}% ({int(ret):>5}/{supp_n})  "
              f"{100*loss/max(1,det):>6.1f}% ({int(loss):>4}/{int(det)})")

    # ── Markdown report ────────────────────────────────────────────────────
    md = []
    md.append("# TP Retention — Baseline Threshold Variant Comparison\n")
    md.append(f"**Dataset**: `ecg_tp_fzark` — clinician-confirmed true positives")
    md.append(f"**Cohort**: {len(df)} records, up to {args.per_class}/event-type "
              f"({'all' if args.per_class == 0 else 'stratified sample'})")
    md.append(f"**Model**: Baseline (`{BASELINE_CKPT}`)")
    md.append(f"**Thresholds**: {', '.join(str(t) for t in BASELINE_THRESHOLD_VARIANTS)}")
    md.append(f"**Suppression**: {'ON' if suppression_enabled else 'OFF'}")
    md.append(f"**Suppressors active**: {sorted(active_events)}\n")
    md.append("All events are clinician-confirmed true positives → "
              "**higher detection & retention is better**.\n")
    md.append("---\n")

    # Baseline default-threshold table
    md.append("## 1. Baseline at default threshold\n")
    md.append("| Event Type | n | Baseline detect% | Baseline retain% |")
    md.append("|---|---|---|---|")
    for _, r in default_df.iterrows():
        md.append(f"| {r['event_type']} | {r['n']} | "
                  f"{r['baseline_detect_pct']:.1f} | {r['baseline_retain_pct']:.1f} |")
    md.append("\n---\n")

    # Threshold variant table
    md.append("## 2. Baseline threshold variant comparison\n")
    hdr = "| Event Type | n |"
    sep = "|---|---|"
    for thr in BASELINE_THRESHOLD_VARIANTS:
        hdr += f" t={thr} det% | t={thr} ret% | t={thr} loss% |"
        sep += "---|---|---|"
    md.append(hdr)
    md.append(sep)
    for _, r in var_df.iterrows():
        line = f"| {r['event_type']} | {r['n']} |"
        for tag in thr_tags:
            det_p  = f"{r[f'{tag}_detect_pct']:.1f}"
            ret_p  = f"{r[f'{tag}_retain_pct']:.1f}"
            loss_p = f"{r[f'{tag}_supp_loss_pct']:.1f}"
            line += f" {det_p} | {ret_p} | {loss_p} |"
        md.append(line)

    md.append("\n---\n")

    # Aggregate tables
    md.append("## 3. Aggregate TP retention\n")
    md.append("### All classes\n")
    md.append("| Threshold | Detection rate | Retention rate | Suppressor TP loss |")
    md.append("|---|---|---|---|")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        det  = var_df[f'{tag}_detected'].sum()
        ret  = var_df[f'{tag}_retained'].sum()
        loss = det - ret
        md.append(f"| {thr} | {100*det/tot_n:.1f}% ({int(det)}/{tot_n}) "
                  f"| **{100*ret/tot_n:.1f}%** ({int(ret)}/{tot_n}) "
                  f"| {100*loss/max(1,det):.1f}% ({int(loss)}/{int(det)}) |")

    md.append(f"\n### Suppressed classes only (n={supp_n})\n")
    md.append("| Threshold | Detection rate | Retention rate | Suppressor TP loss |")
    md.append("|---|---|---|---|")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        det  = supp_v[f'{tag}_detected'].sum()
        ret  = supp_v[f'{tag}_retained'].sum()
        loss = det - ret
        md.append(f"| {thr} | {100*det/supp_n:.1f}% ({int(det)}/{supp_n}) "
                  f"| **{100*ret/supp_n:.1f}%** ({int(ret)}/{supp_n}) "
                  f"| {100*loss/max(1,det):.1f}% ({int(loss)}/{int(det)}) |")

    # Per-class summary for the four suppressed native heads
    md.append("\n---\n")
    md.append("## 4. Per-class retention (native-class heads with suppressors)\n")
    md.append("| Class | n | t=0.5 det→ret | t=0.6 det→ret | t=0.7 det→ret |")
    md.append("|---|---|---|---|---|")
    native_classes = sorted(ACTIVE_RULES.keys())
    for cls in native_classes:
        r = var_df[var_df['event_type'] == cls]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = []
        for tag in thr_tags:
            det_p = r[f'{tag}_detect_pct']
            ret_p = r[f'{tag}_retain_pct']
            cells.append(f"{det_p:.1f}% → **{ret_p:.1f}%**")
        md.append(f"| {cls} | {r['n']} | {' | '.join(cells)} |")

    md.append("\n---\n")
    md.append("## 5. Key observations\n")
    md.append("*(Populated after run — see console output for details.)*\n")
    md.append(f"\nFull CSV: `{var_csv}`")
    md.append(f"\n*Generated from `ecg_tp_fzark` using the baseline model.*")

    md_path = os.path.join(OUT_DIR, f'tp_fzark_threshold_variants_report{out_suffix}.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    # Also save the default-threshold per-class summary
    default_csv = os.path.join(OUT_DIR, f'tp_baseline_default_threshold{out_suffix}.csv')
    default_df.to_csv(default_csv, index=False)

    print(f"\nReport  → {md_path}")
    print(f"CSV     → {var_csv}")
    print(f"Default → {default_csv}")


if __name__ == '__main__':
    main()
