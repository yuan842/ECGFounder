"""Retrain the production L1 ScopeProjection on the fzark TP/FP cohorts.

Frozen backbone + frozen L2. Labels per fzark record (one event type each):
  TP record, scope event  → one-hot scope head = 1
  TP record, out-of-scope → all-zero (real but non-scope → negative)
  FP record (any event)   → all-zero (clinician-removed false alarm → hard negative)

Split by Source Data File (group) → train/val/test, no recording leakage.
Trains 150→6 projection (BCE + pos_weight + temperature), fits per-head max-F1
thresholds (≤5% sens loss vs base) on val, saves res/scope_overlay/fzarkSL.pth,
and evaluates on the held-out fzark test split (TP sensitivity, FP retention).

NOTE: fzark TP has 0 Sinus-Tachycardia → head 6 has no positives (untrainable
here; reported n_pos=0). Runs/Pause DO have positives (V-Run 5212, Pause 82).

Run:  python3 -m scripts.train_l1_fzark
"""
from __future__ import annotations
import os, sys, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from preprocessing import ECGPreprocessor
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS

SCOPE_EVENT_TO_HEAD = {"Atrial Fibrillation": 5, "Bradycardia": 4, "Sinus Tachycardia": 6,
                       "Supraventricular Run": 93, "Ventricular Run": 98, "Pause": 142}
COHORTS = {"TP": "data/ecg_tp_fzark", "FP": "data/ecg_fp_doctor removed1"}
CAP_SCOPE_TP, CAP_FP, CAP_OOS = 3000, 2000, 500
SEED = 42
FEAT = "res/scope_overlay/feat_fzark_train"
OUT_CKPT = "res/scope_overlay/fzarkSL.pth"
DEVICE_CONFIG = "res/scope_overlay/device_configs/fzark.json"
MARGIN = 0.02         # val→test safety margin on both caps
MAX_SENS_LOSS = 0.04  # sensitivity must not drop >4pp below the base head (all heads)

# FZARK threshold policy (per head, on fzark val):
#   τ = min(τ_fp, τ_sens)
#     τ_fp   = lowest threshold keeping FP-retention < the per-head cap
#     τ_sens = highest threshold keeping sensitivity loss < MAX_SENS_LOSS vs base
#   → meets the FP cap when compatible with the 4% sens budget; when they conflict
#     (e.g. AFib's tight cap), the sens budget wins and FP may exceed the cap.
# Heads with 0 device events (Sinus-Tachy 6 on fzark) are NOT listed → base @0.5.
FZARK_FP_LIMITS = {4: 0.10, 5: 0.05, 93: 0.40, 98: 0.05, 142: 0.40}


def build_cohort():
    rows = []
    for cohort, d in COHORTS.items():
        df = pd.read_csv(f"{d}/summary.csv")
        for ev, g in df.groupby("Event Type"):
            scope = ev in SCOPE_EVENT_TO_HEAD
            cap = (CAP_SCOPE_TP if (cohort == "TP" and scope) else
                   CAP_FP if cohort == "FP" else CAP_OOS)
            g = g.sample(n=min(cap, len(g)), random_state=SEED)
            for _, r in g.iterrows():
                y = np.zeros(len(SCOPE_HEADS), np.float32)
                if cohort == "TP" and scope:
                    y[SCOPE_HEADS.index(SCOPE_EVENT_TO_HEAD[ev])] = 1.0
                rows.append((cohort, ev,
                             os.path.join(d, str(r["JSON File"]).replace("\\", "/")),
                             str(r.get("Source Data File", "")), y))
    return pd.DataFrame(rows, columns=["cohort", "event", "path", "src", "y"])


def split_of(src):
    h = int(hashlib.md5(src.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def backbone_logits(df, backbone, device):
    os.makedirs(FEAT, exist_ok=True)
    cache = f"{FEAT}/logits.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(df):
        return np.load(cache)
    prep = ECGPreprocessor.for_fzark()
    xs = []
    for p in tqdm(df["path"], desc="preprocess"):
        try:
            xs.append(prep.from_fzark_json(p))
        except Exception:
            xs.append(torch.zeros(1, 5000))
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 128):
            out.append(backbone(torch.stack(xs[i:i+128]).to(device)).cpu().numpy())
    logits = np.concatenate(out, 0).astype(np.float32)
    np.save(cache, logits)
    return logits


def fit_fp_limited(p, b, df_rows, Y):
    """FZARK policy: τ = min(τ_fp, τ_sens) — FP cap AND ≤4pp sens loss vs base.

    p        : (N,6) L1 probs on val ; b : (N,150) base probs on val
    df_rows  : val-split DataFrame (cohort, event) aligned to p
    Y        : (N,6) scope labels on val
    Per head h with a cap and ≥1 val TP positive:
      τ_fp   = quantile(FP_val_scores, 1 − (cap − MARGIN))      # meet FP cap
      τ_sens = quantile(TP_val_scores, 1 − (base_sens − MAX_SENS_LOSS + MARGIN))
               (vacuous when base_sens ≈ 0 — silent heads have no sens to lose)
      τ      = min(τ_fp, τ_sens)   → FP cap when compatible; sens budget wins on conflict.
    Returns {head: dict(tau, sens, fp_ret, base_sens, binding, fp_cap_met)}.
    """
    cohort = df_rows["cohort"].values
    event = df_rows["event"].values
    out = {}
    for j, h in enumerate(SCOPE_HEADS):
        if h not in FZARK_FP_LIMITS or int(Y[:, j].sum()) == 0:
            continue                      # no cap or 0 device positives → base
        evs = [e for e, hh in SCOPE_EVENT_TO_HEAD.items() if hh == h]
        fp = p[(cohort == "FP") & np.isin(event, evs), j]
        if len(fp) == 0:
            continue
        cap = FZARK_FP_LIMITS[h]
        tau_fp = float(np.quantile(fp, 1.0 - max(0.005, cap - MARGIN)))
        y = Y[:, j].astype(bool)
        base_sens = float((b[y, h] >= 0.5).mean())
        floor = base_sens - MAX_SENS_LOSS
        if floor <= 0:                    # silent base head → no sens to lose
            tau_sens = 0.99
        else:
            tau_sens = float(np.quantile(p[y, j], 1.0 - min(0.999, floor + MARGIN)))
        tau = float(np.clip(min(tau_fp, tau_sens), 0.01, 0.99))
        sens = float((p[y, j] >= tau).mean())
        ret = float((fp >= tau).mean())
        out[h] = dict(tau=tau, sens=sens, fp_ret=ret, base_sens=base_sens,
                      binding=("fp_cap" if tau_fp <= tau_sens else "sens_budget"),
                      fp_cap_met=bool(ret <= cap + 1e-9))
    return out


def main():
    device = resolve_device(); torch.manual_seed(0)
    backbone = load_ecgfounder(device); backbone.eval()
    df = build_cohort()
    df["split"] = df["src"].map(split_of)
    logits = backbone_logits(df, backbone, device)
    Y = np.stack(df["y"].values)
    base = 1/(1+np.exp(-logits))
    sp = df["split"].values
    tr, va, te = sp == "train", sp == "val", sp == "test"
    print(f"rows={len(df)} train={tr.sum()} val={va.sum()} test={te.sum()}")
    print("scope positives (train):", {h: int(Y[tr][:, j].sum()) for j, h in enumerate(SCOPE_HEADS)})

    Xtr = torch.tensor(logits[tr]); Ytr = torch.tensor(Y[tr])
    Xva = torch.tensor(logits[va]); Yva = torch.tensor(Y[va])
    pos = Ytr.sum(0); pw = torch.where(pos > 0, (len(Ytr)-pos)/pos.clamp(min=1), torch.ones_like(pos))
    proj = ScopeProjection()
    opt = torch.optim.Adam(proj.proj.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, bad = -1, None, 0
    for ep in range(300):
        proj.train(); opt.zero_grad(); loss = lossf(proj.proj(Xtr), Ytr); loss.backward(); opt.step()
        proj.eval()
        with torch.no_grad():
            pv = torch.sigmoid(proj.proj(Xva)).numpy()
        prs = [average_precision_score(Yva[:, j], pv[:, j]) for j in range(len(SCOPE_HEADS)) if Yva[:, j].sum() > 0]
        m = float(np.mean(prs))
        if m > best:
            best, best_state, bad = m, {k: v.clone() for k, v in proj.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 25:
            break
    proj.load_state_dict(best_state)
    # temperature per head (grid) on val
    with torch.no_grad():
        zv = proj.proj(Xva)
    temps = torch.ones(len(SCOPE_HEADS))
    for j in range(len(SCOPE_HEADS)):
        if Yva[:, j].sum() == 0:
            continue
        bl = 1e9
        for t in torch.linspace(0.5, 3.0, 26):
            l = nn.functional.binary_cross_entropy_with_logits(zv[:, j]/t, Yva[:, j]).item()
            if l < bl:
                bl, temps[j] = l, float(t)
    proj.temperature.copy_(temps)
    with torch.no_grad():
        pva = proj(Xva).numpy()
    fit = fit_fp_limited(pva, base[va], df[va].reset_index(drop=True), Y[va])
    thr = {h: v["tau"] for h, v in fit.items()}
    dth = proj.decision_threshold.clone()
    for h, t in thr.items():
        dth[SCOPE_HEADS.index(h)] = t
    proj.decision_threshold.copy_(dth)
    torch.save(proj.state_dict(), OUT_CKPT)
    print(f"saved {OUT_CKPT} | thr={ {h: round(v,3) for h,v in thr.items()} } temps={temps.numpy().round(2)}")
    print(f"val fit (τ=min(τ_fp,τ_sens); sens loss budget {MAX_SENS_LOSS:.0%}):")
    for h, v in fit.items():
        capflag = "cap-met" if v["fp_cap_met"] else f"cap-EXCEEDED(>{FZARK_FP_LIMITS[h]})"
        print(f"  head {h:>3} {L.DETECTION_SCOPE[h]:<22} τ={v['tau']:.3f} sens={v['sens']:.3f} "
              f"fp_ret={v['fp_ret']:.3f}/<{FZARK_FP_LIMITS[h]} base_sens={v['base_sens']:.3f} "
              f"binding={v['binding']} {capflag}")

    # ── test eval: TP sensitivity + FP retention, base vs fzark-L1 ──
    with torch.no_grad():
        pte = proj(torch.tensor(logits[te])).numpy()
    dft = df[te].reset_index(drop=True); Yte = Y[te]; bte = base[te]
    print("\n=== fzark TEST: per-head base→fzarkL1 (TP sens ; FP retention) ===")
    print(f"{'head':<26}{'TPn':>6}{'sens b→L1':>16}{'FPn':>6}{'retention b→L1':>18}  cap")
    ev2h = SCOPE_EVENT_TO_HEAD
    test_metrics = {}
    for j, h in enumerate(SCOPE_HEADS):
        tp_m = (Yte[:, j] == 1)
        ev = [e for e, hh in ev2h.items() if hh == h]
        fp_m = (dft["cohort"].values == "FP") & dft["event"].isin(ev).values
        thr_h = thr.get(h, 0.5)
        sens_b = float((bte[tp_m, h] >= 0.5).mean()) if tp_m.sum() else float("nan")
        sens_l = float((pte[tp_m, j] >= thr_h).mean()) if tp_m.sum() else float("nan")
        ret_b = float((bte[fp_m, h] >= 0.5).mean()) if fp_m.sum() else float("nan")
        ret_l = float((pte[fp_m, j] >= thr_h).mean()) if fp_m.sum() else float("nan")
        cap = FZARK_FP_LIMITS.get(h)
        flag = "" if (cap is None or ret_l != ret_l) else (" OK" if ret_l < cap else " ** OVER **")
        print(f"{h} {L.DETECTION_SCOPE[h][:20]:<22}{int(tp_m.sum()):>6}"
              f"  {sens_b:.3f}→{sens_l:.3f}{int(fp_m.sum()):>6}   {ret_b:.3f}→{ret_l:.3f}"
              f"  <{cap}{flag}")
        test_metrics[h] = dict(tp_n=int(tp_m.sum()), fp_n=int(fp_m.sum()),
                               test_sens=sens_l, test_fp_retention=ret_l)

    # ── write the single device config (per-head source/threshold/limit) ──
    import json
    os.makedirs(os.path.dirname(DEVICE_CONFIG), exist_ok=True)
    heads_cfg = {}
    for h in SCOPE_HEADS:
        if h in thr:                      # fzark-L1 routed (has positives + limit)
            heads_cfg[str(h)] = dict(
                name=L.DETECTION_SCOPE[h], source="fzark-L1",
                threshold=round(float(thr[h]), 4), fp_limit=FZARK_FP_LIMITS[h],
                binding=fit[h]["binding"], fp_cap_met=fit[h]["fp_cap_met"],
                test_sens=round(test_metrics[h]["test_sens"], 4),
                test_fp_retention=round(test_metrics[h]["test_fp_retention"], 4))
        else:                             # 0 device events → base @0.5 (global rule)
            heads_cfg[str(h)] = dict(
                name=L.DETECTION_SCOPE[h], source="base", threshold=0.5,
                fp_limit=None, note="0 device events → base (recalibrate per new device)")
    config = dict(
        device="fzark",
        l1_checkpoint=OUT_CKPT,
        policy="τ=min(τ_fp,τ_sens): per-head FP-retention < cap AND sens loss < "
               f"{MAX_SENS_LOSS:.0%} vs base (fzark val); 0-event heads → base@0.5",
        max_sens_loss=MAX_SENS_LOSS, margin=MARGIN,
        scope_heads=SCOPE_HEADS, heads=heads_cfg)
    with open(DEVICE_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nwrote device config → {DEVICE_CONFIG}")


if __name__ == "__main__":
    main()
