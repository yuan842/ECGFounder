"""Comprehensive production evaluation on the fzark TP/FP held-out test split.

Pipeline = fzark device config (fzarkSL routed + per-head FP-cap/4%-sens
thresholds; Sinus-Tachy → base@0.5) → L2 (GT-matched, default ON).
Reuses the cached fzark logits + the seed-fixed split from train_l1_fzark.

Per scope head, on the held-out TEST split, reports base@0.5 vs production:
  TP cohort (event present)        → sensitivity
  FP cohort (clinician-removed)    → FP-retention, specificity (=1−retention)
  pooled TP-vs-FP                  → PPV, F1  (relative — cohorts are
                                     separately capped-sampled, not natural prevalence)
Also reports production WITHOUT L2 to isolate the arbiter's effect.

Writes res/scope_overlay/FZARK_COMPREHENSIVE.md.
Run:  python3 -m scripts.eval_fzark_comprehensive
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from overlay.device_config import load_device_config
from overlay.arbiter import arbitrate, ArbiterConfig
from overlay.types import ScopeScores
from scripts.train_l1_fzark import build_cohort, split_of, SCOPE_EVENT_TO_HEAD, FEAT, FZARK_FP_LIMITS

DEVCFG = "res/scope_overlay/device_configs/fzark.json"
OUT = "res/scope_overlay/FZARK_COMPREHENSIVE.md"


def decisions(probs_row, fire, l2):
    cfg = ArbiterConfig(enabled=l2, fire_threshold=fire)
    d = arbitrate(ScopeScores(probs=probs_row, nsr_score=probs_row.get(1, 0.0)), cfg)
    return {h: d[h].fired for h in SCOPE_HEADS}


def panel(fired, tp_m, fp_m):
    """sens (on TP), fp_retention/spec (on FP), pooled PPV/F1/NPV."""
    tp = int((fired & tp_m).sum()); fn = int((~fired & tp_m).sum())
    fpf = int((fired & fp_m).sum()); tn = int((~fired & fp_m).sum())
    npos = tp + fn
    sens = tp / npos if npos else float("nan")
    fp_ret = fpf / (fpf + tn) if (fpf + tn) else float("nan")
    spec = 1 - fp_ret if fp_ret == fp_ret else float("nan")
    ppv = tp / (tp + fpf) if (tp + fpf) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv == ppv and sens == sens and ppv + sens) else float("nan")
    return dict(sens=sens, fp_ret=fp_ret, spec=spec, ppv=ppv, npv=npv, f1=f1, npos=npos, nneg=fpf + tn)


def main():
    df = build_cohort()
    df["split"] = df["src"].map(split_of)
    logits = np.load(f"{FEAT}/logits.npy")
    assert len(logits) == len(df), "cohort/cache mismatch — re-run train_l1_fzark"
    te = (df["split"] == "test").values
    df_te = df[te].reset_index(drop=True)
    base = 1 / (1 + np.exp(-logits[te]))
    Y = np.stack(df["y"].values)[te]

    proj = ScopeProjection(); proj.load_state_dict(torch.load("res/scope_overlay/fzarkSL.pth", map_location="cpu")); proj.eval()
    dc = load_device_config(DEVCFG)
    l1_heads = set(dc.l1_heads()); fire = {h: 0.5 for h in SCOPE_HEADS}; fire.update(dc.fire_thresholds())
    with torch.no_grad():
        l1p = proj(torch.tensor(logits[te], dtype=torch.float32)).numpy()

    # routed probs per test record
    routed = []
    for r in range(te.sum()):
        routed.append({h: (float(l1p[r, j]) if h in l1_heads else float(base[r, h]))
                       for j, h in enumerate(SCOPE_HEADS)} | {1: float(base[r, 1])})

    # three decision sets
    base_dec = np.array([[base[r, h] >= 0.5 for h in SCOPE_HEADS] for r in range(te.sum())])
    prod_noL2 = np.array([list(decisions(routed[r], fire, l2=False).values()) for r in range(te.sum())])
    prod_L2 = np.array([list(decisions(routed[r], fire, l2=True).values()) for r in range(te.sum())])

    cohort = df_te["cohort"].values; event = df_te["event"].values
    rows = []
    for j, h in enumerate(SCOPE_HEADS):
        evs = [e for e, hh in SCOPE_EVENT_TO_HEAD.items() if hh == h]
        tp_m = (Y[:, j] == 1)
        fp_m = (cohort == "FP") & np.isin(event, evs)
        mb = panel(base_dec[:, j], tp_m, fp_m)
        mn = panel(prod_noL2[:, j], tp_m, fp_m)
        ml = panel(prod_L2[:, j], tp_m, fp_m)
        rows.append((h, dc.heads[h], mb, mn, ml))

    # ── report ──
    md = ["# fzark TP/FP — comprehensive production evaluation",
          "\nProduction = fzark device config (fzarkSL routed + per-head FP-cap/4%-sens "
          "thresholds; Sinus-Tachy→base@0.5) → L2 (GT-matched, default ON). Held-out fzark "
          "TEST split (seed-fixed, grouped by Source Data File). TP→sensitivity; "
          "FP→retention/specificity; pooled PPV/F1 are **relative** (TP/FP cohorts are "
          "separately capped-sampled, not natural prevalence). `prod` = full pipeline (L2 on).\n"]
    md.append("| head | source | TP n | FP n | metric | base | prod(no L2) | **prod (L2)** |")
    md.append("|---|---|---:|---:|---|---:|---:|---:|")
    for h, cfg, mb, mn, ml in rows:
        src = cfg["source"]
        nm = L.DETECTION_SCOPE[h]
        first = True
        for met, lab in (("sens", "Sensitivity"), ("spec", "Specificity"),
                         ("fp_ret", "FP-retention"), ("ppv", "PPV*"), ("f1", "F1*")):
            head_cell = f"{h} {nm}" if first else ""
            src_cell = src if first else ""
            tpn = mb["npos"] if first else ""
            fpn = mb["nneg"] if first else ""
            def fmt(m): return "—" if m[met] != m[met] else f"{m[met]:.3f}"
            md.append(f"| {head_cell} | {src_cell} | {tpn} | {fpn} | {lab} | {fmt(mb)} | {fmt(mn)} | **{fmt(ml)}** |")
            first = False

    with open(OUT, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
