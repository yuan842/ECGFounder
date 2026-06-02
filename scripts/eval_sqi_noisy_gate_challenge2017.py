"""Calibration analysis for a NOISY-only SQI gate (proposal support — not wired in).

Goal: flag Noisy (~) windows while passing clean AFib + Normal. We do NOT try to
separate AFib from Normal — this is a pure signal-quality gate. Evaluated on
Challenge 2017 (A/N = clean, ~ = noisy) at two candidate gating stages (S0 raw,
S3 baseline-removed), reusing the stage taps from eval_sqi_stages_challenge2017.

Reports, per candidate feature and for a combined OR-gate:
  • discrimination AUC (Noisy vs clean)
  • Noisy flag-rate (recall) and clean false-flag-rate at thresholds tied to the
    clean-cohort percentile cut-offs.
Prints a table; writes nothing to production. Run:
  python3 -m scripts.eval_sqi_noisy_gate_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sqi
from scripts.eval_sqi_stages_challenge2017 import (
    center_window, stage_signals, WIN0, GROUPS, DATA, SPLIT, CAP, SEED)
from preprocessing import ECGPreprocessor

# feature: (key, direction) — "lo" = low is noisy, "hi" = high is noisy
FEATS = [("snr_proxy", "lo"), ("kurt", "lo"), ("baseline_drift", "hi"),
         ("hf_noise_ratio", "hi"), ("sqi_score_ecg", "lo")]
STAGE = "S0 raw"          # primary gating stage (pre-compute reject)


def collect(stage):
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    prep = ECGPreprocessor(powerline_hz=60, normalize="winsorize")
    rows = {g: {k: [] for k, _ in FEATS} for g in GROUPS}
    for g in GROUPS:
        ids = sp[sp.label_str == g].idx.values
        rng = np.random.RandomState(SEED)
        ids = ids if len(ids) <= CAP else rng.choice(ids, CAP, replace=False)
        for rid in ids:
            w = center_window(data[rid], WIN0)
            if np.std(w) < 1e-9:
                continue
            sig, fs = dict((s, (x, f)) for s, x, f in stage_signals(w, prep))[stage]
            p = sqi.compute_sqi(sig, fs)
            for k, _ in FEATS:
                rows[g][k].append(p.get(k, np.nan))
    return rows


def arr(rows, g, k):
    v = np.array(rows[g][k], float)
    return v[np.isfinite(v)]


def main():
    rows = collect(STAGE)
    nA, nN, nX = (len(arr(rows, g, "snr_proxy")) for g in ("A", "N", "~"))
    print(f"stage={STAGE}  AFib={nA} Normal={nN} Noisy={nX}\n")

    # ── 1) single-feature discrimination (Noisy vs clean) ──
    print("Single-feature discrimination (Noisy vs clean AFib+Normal):")
    print(f"  {'feature':<18}{'AUC':>7}   clean p-cuts (10/50/90)      noisy p-cuts (10/50/90)")
    feat_auc = {}
    for k, d in FEATS:
        clean = np.concatenate([arr(rows, "A", k), arr(rows, "N", k)])
        noisy = arr(rows, "~", k)
        score = noisy_dir = None
        # AUC: label noisy=1; orient so higher score = more noisy
        y = np.r_[np.zeros(len(clean)), np.ones(len(noisy))]
        s = np.r_[clean, noisy]
        if d == "lo":
            s = -s
        auc = roc_auc_score(y, s)
        feat_auc[k] = (auc, d)
        cp = np.percentile(clean, [10, 50, 90]); npc = np.percentile(noisy, [10, 50, 90])
        print(f"  {k:<18}{auc:>7.3f}   {cp[0]:>8.3g} {cp[1]:>8.3g} {cp[2]:>8.3g}   "
              f"{npc[0]:>8.3g} {npc[1]:>8.3g} {npc[2]:>8.3g}")

    # ── 2) threshold each feature at clean cut-offs; measure rates ──
    print("\nPer-feature gate (threshold at clean cut-off) — flag-rates:")
    print(f"  {'feature @ clean-cut':<28}{'thr':>9}{'Noisy✓':>9}{'AFib✗':>8}{'Norm✗':>8}{'clean✗':>8}")
    def rate(v, thr, d):
        return float(np.mean(v < thr) if d == "lo" else np.mean(v > thr))
    # for lo-features use clean p10/p15/p20 (reject low); for hi use p80/85/90
    cut_for = {"lo": [20, 15, 10], "hi": [80, 85, 90]}
    best_single = None
    for k, d in FEATS:
        clean = np.concatenate([arr(rows, "A", k), arr(rows, "N", k)])
        for q in cut_for[d]:
            thr = float(np.percentile(clean, q))
            nz = rate(arr(rows, "~", k), thr, d)
            fa = rate(arr(rows, "A", k), thr, d); fn = rate(arr(rows, "N", k), thr, d)
            fc = rate(clean, thr, d)
            print(f"  {k+' @p'+str(q):<28}{thr:>9.3g}{nz:>9.1%}{fa:>8.1%}{fn:>8.1%}{fc:>8.1%}")

    # ── 3) combined OR-gate operating points ──
    # Build per-feature boolean at chosen thresholds, OR them, tune to clean budget.
    print("\nCombined OR-gate operating points (Noisy recall @ clean false-flag budget):")
    # candidate gate: SNR<τs OR baseline_drift>τb OR hf_noise>τh OR kurt<τk
    clean_idx = {k: np.concatenate([arr(rows, "A", k), arr(rows, "N", k)]) for k, _ in FEATS}

    def gate_mask(group_key_to_vals, thrs):
        # group_key_to_vals: dict feature->array (same length, per-record)
        m = np.zeros(len(next(iter(group_key_to_vals.values()))), bool)
        for (k, d) in [("snr_proxy", "lo"), ("baseline_drift", "hi"),
                       ("hf_noise_ratio", "hi"), ("kurt", "lo")]:
            v = group_key_to_vals[k]
            m |= (v < thrs[k]) if d == "lo" else (v > thrs[k])
        return m

    # align per-record arrays within a group (drop any NaN rows jointly)
    def grouped(g):
        keys = ["snr_proxy", "baseline_drift", "hf_noise_ratio", "kurt"]
        mat = np.vstack([np.array(rows[g][k], float) for k in keys])
        ok = np.all(np.isfinite(mat), axis=0)
        return {k: mat[i, ok] for i, k in enumerate(keys)}

    gA, gN, gX = grouped("A"), grouped("N"), grouped("~")
    cl = {k: np.concatenate([gA[k], gN[k]]) for k in gA}

    # 4-feature OR (shows over-flagging from weak/inverted features)
    presets = {
        "4-feat conservative": dict(snr_proxy=2, baseline_drift=98, hf_noise_ratio=98, kurt=2),
        "4-feat balanced":     dict(snr_proxy=10, baseline_drift=95, hf_noise_ratio=95, kurt=10),
        "4-feat aggressive":   dict(snr_proxy=20, baseline_drift=90, hf_noise_ratio=90, kurt=20),
    }
    print(f"  {'preset':<22}{'Noisy✓':>9}{'AFib✗':>8}{'Norm✗':>8}{'clean✗':>8}")
    for name, qd in presets.items():
        thrs = {k: float(np.percentile(cl[k], qd[k])) for k in qd}
        nz = gate_mask(gX, thrs).mean(); fa = gate_mask(gA, thrs).mean()
        fn = gate_mask(gN, thrs).mean(); fc = gate_mask(cl, thrs).mean()
        print(f"  {name:<22}{nz:>9.1%}{fa:>8.1%}{fn:>8.1%}{fc:>8.1%}")

    # ── 4) RECOMMENDED 2-feature gate: SNR-low OR drift-high (drop hf/kurt) ──
    print("\n2-feature gate  SNR<τs OR baseline_drift>τb  (recommended form):")
    print(f"  {'operating point':<22}{'τ SNR':>8}{'τ drift':>9}{'Noisy✓':>9}{'AFib✗':>8}{'Norm✗':>8}{'clean✗':>8}")
    def two_mask(g, ts, tb):
        return (g["snr_proxy"] < ts) | (g["baseline_drift"] > tb)
    for name, qsnr, qdrift in [("conservative", 5, 95), ("balanced", 10, 90),
                                ("sensitive", 15, 85), ("aggressive", 20, 80)]:
        ts = float(np.percentile(cl["snr_proxy"], qsnr))
        tb = float(np.percentile(cl["baseline_drift"], qdrift))
        nz = two_mask(gX, ts, tb).mean(); fa = two_mask(gA, ts, tb).mean()
        fn = two_mask(gN, ts, tb).mean(); fc = two_mask(cl, ts, tb).mean()
        print(f"  {name:<22}{ts:>8.3g}{tb:>9.3g}{nz:>9.1%}{fa:>8.1%}{fn:>8.1%}{fc:>8.1%}")


if __name__ == "__main__":
    main()
