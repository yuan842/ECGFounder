"""Production (L1 fuzzySL + L2 default-on) evaluation on the fzark TP & FP cohorts.

fzark TP = confirmed true events  → measures SENSITIVITY (do we keep true detections).
fzark FP = clinician-removed false alarms → measures FALSE-ALARM RETENTION
           (do we suppress them; lower is better → specificity proxy).
Per scope event {AFib 5, Brady 4, Sinus-Tachy 6, SV-Run 93, V-Run 98, Pause 142},
compares baseline (raw head @0.5) vs production (routed L1 + per-head policy
thresholds + GT-matched L2). External validation: thresholds were calibrated on
PTB-XL fuzzy, fzark is a different (ambulatory single-lead) distribution.

Backbone is run fresh on a seed-fixed sample (cap/event) and cached.
Writes res/scope_overlay/PROD_L2_FZARK.md.
Run:  python3 -m scripts.eval_prod_l2_fzark
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from preprocessing import ECGPreprocessor
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, DEFAULT_CKPT
from overlay.inference import L1_HEADS
from overlay.arbiter import arbitrate, ArbiterConfig
from overlay.types import ScopeScores

SCOPE_EVENTS = {"Atrial Fibrillation": 5, "Bradycardia": 4, "Sinus Tachycardia": 6,
                "Supraventricular Run": 93, "Ventricular Run": 98, "Pause": 142}
COHORTS = {"TP": "data/ecg_tp_fzark", "FP": "data/ecg_fp_doctor removed1"}
CAP, SEED = 500, 42
FEAT = "res/scope_overlay/feat_fzark"
OUT = "res/scope_overlay/PROD_L2_FZARK.md"


def cohort_df(cohort):
    d = COHORTS[cohort]
    df = pd.read_csv(f"{d}/summary.csv")
    df = df[df["Event Type"].isin(SCOPE_EVENTS)]
    parts = [g.sample(n=min(CAP, len(g)), random_state=SEED) for _, g in df.groupby("Event Type")]
    df = pd.concat(parts).reset_index(drop=True)
    df["head"] = df["Event Type"].map(SCOPE_EVENTS)
    df["path"] = df["JSON File"].astype(str).str.replace("\\", "/", regex=False).map(lambda r: os.path.join(d, r))
    return df


def backbone_logits(df, cohort, backbone, device):
    cache = f"{FEAT}/{cohort}_logits.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(df):
        return np.load(cache)
    prep = ECGPreprocessor.for_fzark()
    xs, keep = [], []
    for p in tqdm(df["path"], desc=f"{cohort} preprocess"):
        try:
            xs.append(prep.from_fzark_json(p)); keep.append(True)
        except Exception:
            xs.append(torch.zeros(1, 5000)); keep.append(False)
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 128):
            b = torch.stack(xs[i:i+128]).to(device)       # (B,1,5000)
            out.append(backbone(b).cpu().numpy())
    logits = np.concatenate(out, 0).astype(np.float32)
    os.makedirs(FEAT, exist_ok=True); np.save(cache, logits)
    np.save(f"{FEAT}/{cohort}_keep.npy", np.array(keep))
    return logits


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(DEFAULT_CKPT, map_location="cpu")); l1.eval()
    thr = l1.thresholds()                     # shipped fuzzy-calibrated per-head τ
    fire = {h: 0.5 for h in SCOPE_HEADS}; fire.update({h: thr[h] for h in L1_HEADS})
    cfg = ArbiterConfig(enabled=True, fire_threshold=fire)   # L2 default ON

    rows = {}
    for cohort in ("TP", "FP"):
        df = cohort_df(cohort)
        logits = backbone_logits(df, cohort, backbone, device)
        base = 1/(1+np.exp(-logits))
        with torch.no_grad():
            l1p = l1(torch.tensor(logits, dtype=torch.float32)).numpy()
        # routed probs + L2 decisions per record
        prodfire = np.zeros((len(df), len(SCOPE_HEADS)), bool)
        for r in range(len(df)):
            pr = {h: (float(l1p[r, j]) if h in L1_HEADS else float(base[r, h]))
                  for j, h in enumerate(SCOPE_HEADS)}
            d = arbitrate(ScopeScores(probs=pr, nsr_score=float(base[r, 1])), cfg)
            prodfire[r] = [d[h].fired for h in SCOPE_HEADS]
        heads = df["head"].values
        rows[cohort] = (df, base, prodfire, heads)

    md = ["# Production (L1 fuzzySL + L2 on) — fzark TP & FP evaluation",
          "\nfzark TP = confirmed events (↑sensitivity good); FP = clinician-removed false "
          "alarms (↓retention good). Baseline = raw head @0.5. Production = routed L1 + "
          "per-head policy thresholds (PTB-XL-fuzzy-calibrated) + GT-matched L2 (default ON). "
          "External validation — fzark is out-of-distribution from the PTB-XL calibration.\n"]

    # TP sensitivity table
    md.append("## fzark TP — sensitivity (per event, baseline → production+L2)\n")
    md.append("| event | head | n | base sens | prod+L2 sens | Δ |")
    md.append("|---|---|---:|---:|---:|---:|")
    df, base, pf, heads = rows["TP"]
    for ev, h in SCOPE_EVENTS.items():
        j = SCOPE_HEADS.index(h); m = heads == h; n = int(m.sum())
        if n == 0:
            md.append(f"| {ev} | {h} | 0 | — | — | — (no TP) |"); continue
        sb = float((base[m, h] >= 0.5).mean()); sp = float(pf[m, j].mean())
        md.append(f"| {ev} | {h} | {n} | {sb:.3f} | {sp:.3f} | {sp-sb:+.3f} |")

    # FP retention table
    md.append("\n## fzark FP — false-alarm retention (per event, baseline → production+L2; lower better)\n")
    md.append("| event | head | n | base retention | prod+L2 retention | Δ (reduction) |")
    md.append("|---|---|---:|---:|---:|---:|")
    df, base, pf, heads = rows["FP"]
    for ev, h in SCOPE_EVENTS.items():
        j = SCOPE_HEADS.index(h); m = heads == h; n = int(m.sum())
        if n == 0:
            md.append(f"| {ev} | {h} | 0 | — | — | — |"); continue
        rb = float((base[m, h] >= 0.5).mean()); rp = float(pf[m, j].mean())
        md.append(f"| {ev} | {h} | {n} | {rb:.3f} | {rp:.3f} | {rp-rb:+.3f} |")

    with open(OUT, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
