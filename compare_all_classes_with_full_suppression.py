"""
compare_all_classes_with_full_suppression.py
============================================

Re-run the per-class FP-rate comparison on `ecg_fp_doctor removed1`,
applying the v2 feature-gate suppressor stack:

    Atrial Fibrillation              → mean_motion <= 5.0 mG
    Bradycardia                      → mean_hr_bpm <= 56.3 bpm
    Supraventricular Trigeminy       → mean_motion >= 15.0 mG
    Ventricular Trigeminy            → mean_motion >= 24.0 AND snr_proxy > 1.2
    All other event types            → pass-through (no suppression)

Compares baseline vs fine-tuned models head-to-head on every event type
present in the dataset.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import medfilt, iirnotch, filtfilt, butter
from scipy.interpolate import interp1d

from net1d import Net1D
from multiclass_fp_suppression import MultiClassFPSuppressor, ACTIVE_RULES
from device_utils import resolve_device

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR        = "./data/ecg_fp_doctor removed1"
SUMMARY_CSV     = "./data/ecg_fp_doctor removed1/summary.csv"
BASELINE_CKPT   = "./checkpoint/1_lead_ECGFounder.pth"
FINETUNED_CKPT  = "./checkpoint/finetuned_afib_model.pth"
OUT_DIR         = "./res/fp_allclass_full_suppression"

# ECGFounder 150-class index used for each native event-type
LABEL_MAP = {
    "Atrial Fibrillation":            5,
    "Isolated Ventricular Beat":      9,
    "Prolonged RR Interval":         81,
    "Ventricular Couplet":           91,
    "Ventricular Run":               99,
    "Pause":                        143,
    "Sinus Tachycardia":              6,
    "Supraventricular Couplet":      20,
    "Isolated Supraventricular Beat":16,
}
AFIB_IDX = 5

MODEL_THRESHOLDS = {
    "baseline":   0.5,
    "finetuned":  0.2646,
}

# Baseline threshold variants for sensitivity analysis
BASELINE_THRESHOLD_VARIANTS = [0.5, 0.6, 0.7]

# Per-class suppression — which event types have an active v2 filter rule
SUPPRESSED_EVENTS = set(ACTIVE_RULES.keys())  # AFib, Bradycardia, SV Trigeminy, V Trigeminy

# Pre-processing constants (match eval_ecg_tprex.py)
FS_IN, FS_OUT  = 128, 500
TARGET_LEN     = 5000
MAGNIFICATION  = 1000
POWERLINE_HZ   = 50
WIN_LIMITS     = (1.5, 98.5)
BATCH_SIZE     = 64


def load_json_ecg(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        records = json.load(f)
    raw = np.concatenate([np.array(r['data']['ecg'], dtype=np.float32)
                          for r in records])
    return raw / MAGNIFICATION

def robust_preprocess(signal_1d: np.ndarray, fs_in: int = FS_IN) -> torch.Tensor:
    x = signal_1d[np.newaxis, :]
    b, a = iirnotch(POWERLINE_HZ, 30, fs_in); x = filtfilt(b, a, x, axis=1)
    b, a = butter(4, [0.67, 40.0], btype='bandpass', fs=fs_in); x = filtfilt(b, a, x, axis=1)
    kernel = int(0.4 * fs_in) | 1
    baseline = medfilt(x[0], kernel_size=kernel); x[0] = x[0] - baseline
    t = x.shape[1] / fs_in
    x_old = np.linspace(0, t, num=x.shape[1], endpoint=True)
    x_new = np.linspace(0, t, num=int(t * FS_OUT), endpoint=True)
    f = interp1d(x_old, x[0], kind='linear', bounds_error=False, fill_value=0.0)
    sig = f(x_new)
    if len(sig) >= TARGET_LEN:
        s = (len(sig) - TARGET_LEN) // 2; sig = sig[s:s + TARGET_LEN]
    else:
        pad = TARGET_LEN - len(sig); sig = np.pad(sig, (pad // 2, pad - pad // 2))
    lo, hi = np.percentile(sig, WIN_LIMITS); sig = np.clip(sig, lo, hi)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return torch.from_numpy(sig.astype(np.float32))[None, :]


class FpDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True); self.data_dir = data_dir
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row['JSON File']).replace('\\', '/')
        path = os.path.join(self.data_dir, rel)
        try:
            sig = load_json_ecg(path); tensor = robust_preprocess(sig)
        except Exception:
            tensor = torch.zeros(1, TARGET_LEN)
        return tensor, idx


def build_model(ckpt_path, device):
    m = Net1D(in_channels=1, base_filters=64, ratio=1,
              filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16, stride=2,
              groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=150)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    m.load_state_dict(sd)
    return m.to(device).eval()


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
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}\n")

    # Stratified sample
    df = pd.read_csv(SUMMARY_CSV)
    sampled = [g.sample(n=min(args.per_class, len(g)), random_state=args.seed)
               for _, g in df.groupby('Event Type')]
    df = pd.concat(sampled).reset_index(drop=True)
    print(f"Sampled cohort: {len(df)} records across {df['Event Type'].nunique()} event types")
    print(df['Event Type'].value_counts().to_string()); print()

    ds = FpDataset(df, DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Run both models ─────────────────────────────────────────────────────
    probs_dict = {}
    for name, ckpt in [('baseline', BASELINE_CKPT), ('finetuned', FINETUNED_CKPT)]:
        print(f"\n{'='*72}\nMODEL: {name.upper()}\n{'='*72}")
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} not found"); continue
        model = build_model(ckpt, device)
        probs = run_inference(model, dl, device)
        np.save(os.path.join(OUT_DIR, f"{name}_probs_full.npy"), probs)
        probs_dict[name] = probs
    if len(probs_dict) < 2:
        print("Need both models — exiting."); return

    # ── Suppressor (multiclass router) ──────────────────────────────────────
    suppressor = MultiClassFPSuppressor()
    print(f"\nSuppressor supports: {suppressor.supported_classes()}\n")

    # Pre-compute suppressor decisions for every event whose event type
    # has an active v2 filter rule.
    print("Pre-computing suppressor decisions...")
    # v2 routing: suppressor is called with the event's own event_type name.
    # No cross-class routing (v1 routed unmapped events through AFib).
    decisions = {}      # (event_idx, model_name) → SuppressionResult
    for i, row in tqdm(df.iterrows(), total=len(df), desc="  suppressor"):
        et = row['Event Type']
        rel = str(row['JSON File']).replace('\\','/')
        path = os.path.join(DATA_DIR, rel)
        # v2: route by event type name directly
        if et not in SUPPRESSED_EVENTS:
            continue   # no rule for this class; alert passes through
        for name, probs in probs_dict.items():
            cls_idx = LABEL_MAP.get(et, AFIB_IDX)
            p = float(probs[i, cls_idx])
            if p < MODEL_THRESHOLDS[name]:
                continue   # not an alert, no need to run suppressor
            try:
                r = suppressor.suppress_alert(et, p, path)
                decisions[(int(i), name)] = r
            except Exception:
                decisions[(int(i), name)] = None

    # ── Per-event-type metrics ──────────────────────────────────────────────
    print("\nCompiling per-class results...")
    rows = []
    for et, grp in df.groupby('Event Type'):
        idxs = grp.index.values
        n_total = len(idxs)
        in_map = et in LABEL_MAP
        cls_idx = LABEL_MAP.get(et, AFIB_IDX)
        head_name = et if in_map else f"{et} → AFib head"

        # v2: route by event type name directly
        has_suppressor = et in SUPPRESSED_EVENTS

        row = dict(event_type=et, n=n_total, in_label_map=in_map,
                   class_idx=cls_idx, eval_head=head_name,
                   suppressor_target=et if has_suppressor else 'none')

        for name, probs in probs_dict.items():
            mt = MODEL_THRESHOLDS[name]
            p_tgt = probs[idxs, cls_idx]
            raw_alerts = int((p_tgt >= mt).sum())
            row[f'{name}_raw_alerts']   = raw_alerts
            row[f'{name}_raw_rate_pct'] = 100 * raw_alerts / max(1, n_total)

            if has_suppressor:
                post_kept = 0
                for pos, ei in enumerate(idxs):
                    if p_tgt[pos] < mt: continue   # not an alert
                    sr = decisions.get((int(ei), name))
                    if sr is None or sr.keep: post_kept += 1
                row[f'{name}_post_alerts']    = int(post_kept)
                row[f'{name}_post_rate_pct']  = 100 * post_kept / max(1, n_total)
                row[f'{name}_supp_gain_pct']  = (100 * (raw_alerts - post_kept) /
                                                  max(1, raw_alerts)) if raw_alerts > 0 else 0.0
            else:
                row[f'{name}_post_alerts']    = None
                row[f'{name}_post_rate_pct']  = None
                row[f'{name}_supp_gain_pct']  = None
        rows.append(row)

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(['in_label_map','event_type'],
                                ascending=[False, True])
    csv_out = os.path.join(OUT_DIR, 'allclass_full_suppression.csv')
    res_df.to_csv(csv_out, index=False)

    # ── Console summary ─────────────────────────────────────────────────────
    print("\n" + "="*108)
    print("PER-CLASS FP COMPARISON  with full per-class suppression  "
          "(lower alert rate = better, all events are FPs)")
    print("="*108)
    print(f"\n{'Event Type':<33} {'n':>4} {'route':>16}  "
          f"{'BS raw%':>7} {'BS post%':>8} | "
          f"{'FT raw%':>7} {'FT post%':>8}  "
          f"{'Δ post':>7}")
    print("-"*108)
    for _, r in res_df.iterrows():
        bs_post = (f"{r['baseline_post_rate_pct']:>8.1f}"
                   if r['baseline_post_rate_pct'] is not None else f"{'—':>8}")
        ft_post = (f"{r['finetuned_post_rate_pct']:>8.1f}"
                   if r['finetuned_post_rate_pct'] is not None else f"{'—':>8}")
        if (r['baseline_post_rate_pct'] is not None and
                r['finetuned_post_rate_pct'] is not None):
            d_post = (f"{r['baseline_post_rate_pct'] - r['finetuned_post_rate_pct']:+7.1f}")
        else:
            d_post = f"{'—':>7}"
        marker = "*" if r['in_label_map'] else " "
        route = (r['suppressor_target'][:14]
                 if r['suppressor_target'] != 'none' else 'no_supp')
        print(f"{r['event_type']:<33} {r['n']:>4} {route:>16}  "
              f"{r['baseline_raw_rate_pct']:>6.1f} {bs_post}  | "
              f"{r['finetuned_raw_rate_pct']:>6.1f} {ft_post}  "
              f"{d_post}{marker}")
    print("-"*108)
    print("* = native-class head | unmarked = AFib-head cross-class evaluation")

    # Aggregate (weighted by n)
    print("\nAggregate across mapped classes (weighted by n):")
    mapped = res_df[res_df['in_label_map']]
    tot_n = mapped['n'].sum()
    for name in probs_dict.keys():
        raw = mapped[f'{name}_raw_alerts'].sum()
        post = mapped.loc[mapped[f'{name}_post_alerts'].notna(),
                          f'{name}_post_alerts'].sum()
        post_n = mapped.loc[mapped[f'{name}_post_alerts'].notna(), 'n'].sum()
        print(f"  {name:>10}  raw FP rate = {100*raw/tot_n:>5.2f}%  ({raw}/{tot_n})  "
              f"|  post FP rate (suppressed classes) = {100*post/post_n:>5.2f}%  "
              f"({post}/{post_n})")

    # Markdown report
    md = []
    md.append("# All-Class FP Comparison with Full Per-Class Suppression\n")
    md.append(f"**Cohort**: {len(df)} records, up to {args.per_class}/event-type")
    md.append(f"**Device**: {device}")
    md.append(f"**Suppressors active**: {sorted(SUPPRESSED_EVENTS)}\n")
    md.append("## Per-class results\n")
    md.append("| Event Type | n | route | Baseline raw% | Baseline post% | Fine-tuned raw% | Fine-tuned post% |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in res_df.iterrows():
        bs_post = f"{r['baseline_post_rate_pct']:.1f}" if r['baseline_post_rate_pct'] is not None else "—"
        ft_post = f"{r['finetuned_post_rate_pct']:.1f}" if r['finetuned_post_rate_pct'] is not None else "—"
        route = r['suppressor_target'] if r['suppressor_target'] != 'none' else '—'
        md.append(f"| {r['event_type']} | {r['n']} | {route} | "
                  f"{r['baseline_raw_rate_pct']:.1f} | {bs_post} | "
                  f"{r['finetuned_raw_rate_pct']:.1f} | {ft_post} |")
    md.append(f"\nFull CSV: `{csv_out}`")
    with open(os.path.join(OUT_DIR, 'allclass_full_suppression_report.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f"\nReport → {OUT_DIR}/allclass_full_suppression_report.md")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE THRESHOLD VARIANT COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    # Re-use baseline probs (already computed) and suppressor decisions
    # (events passing higher thresholds are a subset of threshold=0.5 decisions).
    # For events that are NEW alerts at lower thresholds we need additional
    # suppressor calls — but all three variants are ≥ 0.5 so existing decisions
    # already cover everything.
    baseline_probs = probs_dict['baseline']

    # --- Need suppressor decisions for any event where p >= min(thresholds) ---
    # The original loop used threshold 0.5, so decisions exist for p >= 0.5.
    # All our variants are ≥ 0.5, so no extra suppressor calls needed.

    print("\n\n" + "="*120)
    print("BASELINE THRESHOLD VARIANT COMPARISON")
    print(f"  Thresholds: {BASELINE_THRESHOLD_VARIANTS}")
    print("  (same baseline model, same suppressor stack, varying the alert-fire threshold)")
    print("="*120)

    # Collect per-event-type results for each threshold
    variant_rows = []
    for et, grp in df.groupby('Event Type'):
        idxs = grp.index.values
        n_total = len(idxs)
        in_map = et in LABEL_MAP
        cls_idx = LABEL_MAP.get(et, AFIB_IDX)

        # v2: route by event type name directly
        has_suppressor = et in SUPPRESSED_EVENTS

        vrow = dict(event_type=et, n=n_total, in_label_map=in_map,
                    class_idx=cls_idx,
                    suppressor_target=et if has_suppressor else 'none')

        p_tgt = baseline_probs[idxs, cls_idx]

        for thr in BASELINE_THRESHOLD_VARIANTS:
            tag = f"bs_{thr:.1f}"
            raw_alerts = int((p_tgt >= thr).sum())
            vrow[f'{tag}_raw_alerts']   = raw_alerts
            vrow[f'{tag}_raw_rate_pct'] = 100 * raw_alerts / max(1, n_total)

            if has_suppressor:
                post_kept = 0
                for pos, ei in enumerate(idxs):
                    if p_tgt[pos] < thr:
                        continue  # not an alert at this threshold
                    sr = decisions.get((int(ei), 'baseline'))
                    if sr is None or sr.keep:
                        post_kept += 1
                vrow[f'{tag}_post_alerts']   = int(post_kept)
                vrow[f'{tag}_post_rate_pct'] = 100 * post_kept / max(1, n_total)
                vrow[f'{tag}_supp_gain_pct'] = (100 * (raw_alerts - post_kept)
                                                 / max(1, raw_alerts)) if raw_alerts > 0 else 0.0
            else:
                # No suppressor → post = raw (alert passes through)
                vrow[f'{tag}_post_alerts']   = raw_alerts
                vrow[f'{tag}_post_rate_pct'] = 100 * raw_alerts / max(1, n_total)
                vrow[f'{tag}_supp_gain_pct'] = 0.0
        variant_rows.append(vrow)

    var_df = pd.DataFrame(variant_rows)
    var_df = var_df.sort_values(['in_label_map', 'event_type'], ascending=[False, True])
    var_csv = os.path.join(OUT_DIR, 'baseline_threshold_variants.csv')
    var_df.to_csv(var_csv, index=False)

    # Console table
    thr_tags = [f"bs_{t:.1f}" for t in BASELINE_THRESHOLD_VARIANTS]
    hdr_cols = "  ".join(f"{'raw%':>6} {'post%':>6}" for _ in BASELINE_THRESHOLD_VARIANTS)
    thr_hdr  = "  ".join(f"  t={t:.1f}       " for t in BASELINE_THRESHOLD_VARIANTS)
    print(f"\n{'Event Type':<33} {'n':>4}   {thr_hdr}")
    print(f"{'':33} {'':>4}   {hdr_cols}")
    print("-"*120)
    for _, r in var_df.iterrows():
        parts = []
        for tag in thr_tags:
            raw_pct = r[f'{tag}_raw_rate_pct']
            post_pct = r[f'{tag}_post_rate_pct']
            raw_s = f"{raw_pct:>6.1f}"
            post_s = f"{post_pct:>6.1f}"
            parts.append(f"{raw_s} {post_s}")
        marker = "*" if r['in_label_map'] else " "
        print(f"{r['event_type']:<33} {r['n']:>4}   {'  '.join(parts)} {marker}")
    print("-"*120)
    print("* = native-class head   |  raw% = alerts before suppression  |  post% = after suppression")

    # Aggregate for each threshold
    print("\nAggregate (mapped/suppressed classes, weighted by n):")
    mapped_v = var_df[var_df['in_label_map']]
    tot_n_v  = mapped_v['n'].sum()
    print(f"  {'threshold':>10}  {'raw FP rate':>18}  {'post FP rate':>24}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*24}")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        raw  = mapped_v[f'{tag}_raw_alerts'].sum()
        mask = mapped_v[f'{tag}_post_alerts'].notna()
        post   = mapped_v.loc[mask, f'{tag}_post_alerts'].sum()
        post_n = mapped_v.loc[mask, 'n'].sum()
        print(f"  {thr:>10.1f}  {100*raw/tot_n_v:>6.2f}% ({int(raw):>4}/{tot_n_v})  "
              f"{100*post/post_n:>6.2f}% ({int(post):>4}/{int(post_n)}) [suppressed only]")

    # ── Variant markdown report ────────────────────────────────────────────
    vmd = []
    vmd.append("# Baseline Threshold Variant Comparison\n")
    vmd.append(f"**Cohort**: {len(df)} records, up to {args.per_class}/event-type")
    vmd.append(f"**Model**: Baseline (`{BASELINE_CKPT}`)")
    vmd.append(f"**Thresholds**: {', '.join(str(t) for t in BASELINE_THRESHOLD_VARIANTS)}")
    vmd.append(f"**Suppressors active**: {sorted(SUPPRESSED_EVENTS)}\n")
    vmd.append("All events are clinician-removed false positives → **lower alert rate is better**.\n")
    vmd.append("---\n")

    vmd.append("## Per-class results\n")
    # Build header
    hdr = "| Event Type | n | route |"
    sep = "|---|---|---|"
    for thr in BASELINE_THRESHOLD_VARIANTS:
        hdr += f" t={thr} raw% | t={thr} post% |"
        sep += "---|---|"
    vmd.append(hdr)
    vmd.append(sep)
    for _, r in var_df.iterrows():
        route = r['suppressor_target'] if r['suppressor_target'] != 'none' else '—'
        line = f"| {r['event_type']} | {r['n']} | {route} |"
        for tag in thr_tags:
            raw_pct  = f"{r[f'{tag}_raw_rate_pct']:.1f}"
            post_pct = r[f'{tag}_post_rate_pct']
            post_s   = f"{post_pct:.1f}"
            line += f" {raw_pct} | {post_s} |"
        vmd.append(line)

    vmd.append("\n---\n")
    vmd.append("## Aggregate (mapped/suppressed classes)\n")
    vmd.append("| Threshold | Raw FP rate | Post-suppression FP rate |")
    vmd.append("|---|---|---|")
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        raw  = mapped_v[f'{tag}_raw_alerts'].sum()
        mask = mapped_v[f'{tag}_post_alerts'].notna()
        post   = mapped_v.loc[mask, f'{tag}_post_alerts'].sum()
        post_n = mapped_v.loc[mask, 'n'].sum()
        vmd.append(f"| {thr} | {100*raw/tot_n_v:.1f}% ({int(raw)}/{tot_n_v}) "
                   f"| **{100*post/post_n:.1f}%** ({int(post)}/{int(post_n)}) |")

    # Suppressed-only aggregate
    vmd.append("\n### Suppressed classes only\n")
    vmd.append("| Threshold | Raw FP rate | Post-suppression FP rate |")
    vmd.append("|---|---|---|")
    supp_v = var_df[var_df['suppressor_target'] != 'none']
    supp_n = supp_v['n'].sum()
    for thr, tag in zip(BASELINE_THRESHOLD_VARIANTS, thr_tags):
        raw  = supp_v[f'{tag}_raw_alerts'].sum()
        post = supp_v[f'{tag}_post_alerts'].sum()
        vmd.append(f"| {thr} | {100*raw/supp_n:.1f}% ({int(raw)}/{supp_n}) "
                   f"| **{100*post/supp_n:.1f}%** ({int(post)}/{supp_n}) |")

    vmd.append("\n---\n")

    # Per-class delta table for the four suppressed native heads
    vmd.append("## Per-class post-suppression comparison (native-class heads only)\n")
    vmd.append("| Class | t=0.5 raw→post | t=0.6 raw→post | t=0.7 raw→post |")
    vmd.append("|---|---|---|---|")
    native_classes = sorted(ACTIVE_RULES.keys())
    for cls in native_classes:
        r = var_df[var_df['event_type'] == cls]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = []
        for tag in thr_tags:
            raw_p  = r[f'{tag}_raw_rate_pct']
            post_p = r[f'{tag}_post_rate_pct']
            cells.append(f"{raw_p:.1f}% → **{post_p:.1f}%**")
        vmd.append(f"| {cls} | {' | '.join(cells)} |")

    vmd.append("\n---\n")
    vmd.append("## Key observations\n")
    vmd.append("*(Auto-populated from the data above — see per-class and aggregate tables.)*\n")
    vmd.append(f"**Suppressor version**: v2 feature-gate filters")
    vmd.append(f"**Active rules**: {sorted(ACTIVE_RULES.keys())}")
    vmd.append(f"\n\nFull CSV: `{var_csv}`")
    vmd.append(f"\n*Generated from the same inference run as the main comparison.*")

    var_md_path = os.path.join(OUT_DIR, 'baseline_threshold_variants_report.md')
    with open(var_md_path, 'w') as f:
        f.write('\n'.join(vmd))
    print(f"\nVariant report → {var_md_path}")
    print(f"Variant CSV    → {var_csv}")


if __name__ == '__main__':
    main()
