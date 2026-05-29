"""Combined TP+FP binary-classification metrics.

For each V3.1-mapped event class with both TP (ecg_tp_fzark) and FP
(ecg_fp_doctor removed1) records, compute Sensitivity / Specificity / PPV /
NPV at t=0.5 plus ROC-AUC / PR-AUC, under FP suppression OFF and ON.

Positives = TP records (labeled positive by clinicians).
Negatives = FP records (labeled negative — clinicians removed them).
Score     = sigmoid output at the V3.1-assigned 150-class head.
Under ON  = score is set to 0.0 if the suppressor returned keep=False.

Uses the cached probability matrices written by the OFF-suppression runs
(same record order as the stratified-sample dataframes the eval scripts
build with random_state=42).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from label_config import FZARK_LABEL_MAP, scope_indices, head_threshold
from multiclass_fp_suppression import MultiClassFPSuppressor

OUT_DIR = "res/cross_dataset_supp_off"
TP_DIR  = "data/ecg_tp_fzark"
FP_DIR  = "data/ecg_fp_doctor removed1"


def load_cohort(summary_csv, data_dir, probs_path, seed=42, per_class=500):
    df = pd.read_csv(summary_csv)
    sampled = [g.sample(n=min(per_class, len(g)), random_state=seed)
               for _, g in df.groupby('Event Type')]
    df = pd.concat(sampled).reset_index(drop=True)
    probs = np.load(probs_path)
    assert len(df) == probs.shape[0], (len(df), probs.shape)
    return df, probs


def main():
    print("Loading cached cohorts and probability matrices...")
    tp_df, tp_probs = load_cohort(
        f"{TP_DIR}/summary.csv", TP_DIR,
        "res/tp_fzark_full_suppression/baseline_probs_tp.npy",
    )
    fp_df, fp_probs = load_cohort(
        f"{FP_DIR}/summary.csv", FP_DIR,
        "res/fp_allclass_full_suppression/baseline_probs_full.npy",
    )
    print(f"  TP records: {len(tp_df)}  FP records: {len(fp_df)}")

    suppressor = MultiClassFPSuppressor(enabled=True)

    # Resolve suppression decisions per record.
    # For records whose event_type isn't in ACTIVE_RULES the suppressor
    # passes through; we only call it for records that would actually
    # alert (score >= 0.5) to save JSON I/O.
    def supp_score(et, score, json_rel, data_dir):
        if score < head_threshold(FZARK_LABEL_MAP[et]):
            return score
        # Only AFib / Bradycardia / SV Trig / V Trig have rules
        if et not in ('Atrial Fibrillation', 'Bradycardia',
                      'Supraventricular Trigeminy', 'Ventricular Trigeminy'):
            return score
        path = os.path.join(data_dir, str(json_rel).replace('\\', '/'))
        try:
            r = suppressor.suppress_alert(et, score, path)
            return score if r.keep else 0.0
        except Exception:
            return score

    print("Computing post-suppression scores (calling suppressor where needed)...")
    # Limit to in-scope detection heads (label_config.DETECTION_SCOPE).
    in_scope_events = sorted(e for e in FZARK_LABEL_MAP
                             if FZARK_LABEL_MAP[e] in scope_indices())
    print(f"In-scope events ({len(in_scope_events)}): {in_scope_events}")
    rows = []
    for et in in_scope_events:
        head_idx = FZARK_LABEL_MAP[et]
        thr = head_threshold(head_idx)

        tp_mask = (tp_df['Event Type'] == et).values
        fp_mask = (fp_df['Event Type'] == et).values
        n_tp = int(tp_mask.sum())
        n_fp = int(fp_mask.sum())
        if n_tp == 0 and n_fp == 0:
            continue

        pos_scores_off = tp_probs[tp_mask, head_idx]
        neg_scores_off = fp_probs[fp_mask, head_idx]

        # Suppression-ON scores (only AFib + Bradycardia remain in-scope active;
        # SV-Trig/V-Trig heads are out of DETECTION_SCOPE → suppressor passes through)
        pos_scores_on = pos_scores_off.copy()
        neg_scores_on = neg_scores_off.copy()
        if et in ('Atrial Fibrillation', 'Bradycardia'):
            tp_sub = tp_df[tp_mask].reset_index(drop=True)
            fp_sub = fp_df[fp_mask].reset_index(drop=True)
            for i, row in tqdm(tp_sub.iterrows(), total=len(tp_sub),
                                desc=f"  {et[:25]:<25} TP supp", leave=False):
                pos_scores_on[i] = supp_score(et, pos_scores_on[i],
                                              row['JSON File'], TP_DIR)
            for i, row in tqdm(fp_sub.iterrows(), total=len(fp_sub),
                                desc=f"  {et[:25]:<25} FP supp", leave=False):
                neg_scores_on[i] = supp_score(et, neg_scores_on[i],
                                              row['JSON File'], FP_DIR)

        for mode, ps, ns in [('OFF', pos_scores_off, neg_scores_off),
                              ('ON',  pos_scores_on,  neg_scores_on)]:
            tp_alerts = int((ps >= thr).sum())
            fp_alerts = int((ns >= thr).sum())
            fn        = n_tp - tp_alerts
            tn        = n_fp - fp_alerts

            sens = tp_alerts / max(1, n_tp)
            spec = tn / max(1, n_fp)
            ppv  = tp_alerts / max(1, tp_alerts + fp_alerts)
            npv  = tn / max(1, tn + fn)
            acc  = (tp_alerts + tn) / max(1, n_tp + n_fp)
            f1_denom = 2 * tp_alerts + fp_alerts + fn
            f1   = (2 * tp_alerts) / f1_denom if f1_denom > 0 else 0.0

            # AUCs need both classes present
            if n_tp > 0 and n_fp > 0:
                y = np.concatenate([np.ones(n_tp), np.zeros(n_fp)])
                s = np.concatenate([ps, ns])
                roc_auc = float(roc_auc_score(y, s))
                pr_auc  = float(average_precision_score(y, s))
            else:
                roc_auc, pr_auc = float('nan'), float('nan')

            rows.append(dict(
                event_type=et, head_idx=head_idx, mode=mode,
                n_pos=n_tp, n_neg=n_fp,
                TP=tp_alerts, FP=fp_alerts, FN=fn, TN=tn,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                accuracy=acc, f1=f1,
                roc_auc=roc_auc, pr_auc=pr_auc,
            ))

    per_class = pd.DataFrame(rows)

    # ── Overall (micro) — sum per-class confusion cells (each event already
    # thresholded at its own head_threshold), pool raw scores for AUC only. ──
    # Pooling raw scores then thresholding at a single value would be wrong now
    # that heads use different thresholds, so we sum cells instead.
    both = [et for et in in_scope_events
            if (tp_df['Event Type'] == et).any() and (fp_df['Event Type'] == et).any()]

    overall_rows = []
    for mode in ('OFF', 'ON'):
        sub = per_class[(per_class['mode'] == mode) & (per_class['event_type'].isin(both))]
        tp_alerts = int(sub['TP'].sum()); fp_alerts = int(sub['FP'].sum())
        fn = int(sub['FN'].sum()); tn = int(sub['TN'].sum())
        n_pos = tp_alerts + fn; n_neg = fp_alerts + tn
        # AUC: pool raw OFF scores (threshold-independent; suppression on ON would
        # zero scores, so AUC is reported on the OFF/raw scores for both modes)
        pos_all, neg_all = [], []
        for et in both:
            h = FZARK_LABEL_MAP[et]
            pos_all.append(tp_probs[(tp_df['Event Type'] == et).values, h])
            neg_all.append(fp_probs[(fp_df['Event Type'] == et).values, h])
        y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        s = np.concatenate([np.concatenate(pos_all), np.concatenate(neg_all)])
        f1_denom = 2*tp_alerts + fp_alerts + fn
        overall_rows.append(dict(
            event_type='OVERALL (micro-pooled)', head_idx=-1, mode=mode,
            n_pos=n_pos, n_neg=n_neg,
            TP=tp_alerts, FP=fp_alerts, FN=fn, TN=tn,
            sensitivity=tp_alerts/max(1, n_pos), specificity=tn/max(1, n_neg),
            ppv=tp_alerts/max(1, tp_alerts+fp_alerts),
            npv=tn/max(1, tn+fn),
            accuracy=(tp_alerts+tn)/max(1, n_pos+n_neg),
            f1=(2*tp_alerts)/f1_denom if f1_denom > 0 else 0.0,
            roc_auc=float(roc_auc_score(y, s)),
            pr_auc=float(average_precision_score(y, s)),
        ))

    out = pd.concat([per_class, pd.DataFrame(overall_rows)], ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'combined_binary_metrics_scope.csv')
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Pretty-print
    pd.set_option('display.float_format', lambda x: f"{x:.3f}")
    print("\n" + "="*120)
    print("PER-CLASS BINARY METRICS — in-scope heads, per-head thresholds (OFF vs ON)")
    print("="*120)
    cols = ['event_type', 'mode', 'n_pos', 'n_neg', 'TP', 'FP', 'FN', 'TN',
            'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'f1',
            'roc_auc', 'pr_auc']
    print(out[cols].to_string(index=False))


if __name__ == '__main__':
    main()
