"""Comprehensive eval of the base-routed-head L2 refinement on fzark TP/FP.

Recommendation under test: on a device where a head is base-routed (uncalibrated,
base@0.5), exclude it from the L2 rate-exclusion arbitration — its base@0.5 score
is too unreliable to suppress (or be suppressed by) a calibrated peer. The
alternative is to disable the head outright.

On the fzark device config the only base-routed scope head is **Sinus-Tachy (6)**
(0 device events → base@0.5). It shares the rate-exclusion group {Brady 4, AFib 5,
Tachy 6}, so the refinement can change decisions for head 6 AND its calibrated
peers. This script measures all of it on the held-out fzark TEST split.

Four production variants (all = fzark device cfg + L2 GT-matched, default ON):
  current   uncalibrated head participates in arbitration (pre-refinement behavior)
  exclude   (a) head 6 excluded from rate-exclusion arbitration   ← implemented change
  disable   (b) head 6 disabled outright (never fires)            ← alternative
  base      raw head @0.5 (context only)

Reuses cached fzark logits + the seed-fixed split from train_l1_fzark.
Writes res/scope_overlay/UNCAL_EXCLUSION_FZARK.md.
Run:  python3 -m scripts.eval_uncal_exclusion_fzark
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from overlay.device_config import load_device_config
from overlay.arbiter import arbitrate, ArbiterConfig, BRADY, AFIB, TACHY
from overlay.types import ScopeScores
from scripts.train_l1_fzark import build_cohort, split_of, SCOPE_EVENT_TO_HEAD, FEAT

DEVCFG = "res/scope_overlay/device_configs/fzark.json"
OUT = "res/scope_overlay/UNCAL_EXCLUSION_FZARK.md"
RATE_GROUP = (BRADY, AFIB, TACHY)
NEVER = 1.01   # threshold that no sigmoid prob can reach → head disabled


def decide(routed, fire, uncal):
    cfg = ArbiterConfig(enabled=True, fire_threshold=fire, uncalibrated_heads=uncal)
    d = arbitrate(ScopeScores(probs=routed, nsr_score=routed.get(1, 0.0)), cfg)
    return {h: d[h].fired for h in SCOPE_HEADS}


def panel(fired, tp_m, fp_m):
    tp = int((fired & tp_m).sum()); fn = int((~fired & tp_m).sum())
    fpf = int((fired & fp_m).sum()); tn = int((~fired & fp_m).sum())
    npos = tp + fn
    sens = tp / npos if npos else float("nan")
    fp_ret = fpf / (fpf + tn) if (fpf + tn) else float("nan")
    spec = 1 - fp_ret if fp_ret == fp_ret else float("nan")
    ppv = tp / (tp + fpf) if (tp + fpf) else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv == ppv and sens == sens and ppv + sens) else float("nan")
    return dict(sens=sens, fp_ret=fp_ret, spec=spec, ppv=ppv, f1=f1, npos=npos, nneg=fpf + tn)


def main():
    df = build_cohort()
    df["split"] = df["src"].map(split_of)
    logits = np.load(f"{FEAT}/logits.npy")
    assert len(logits) == len(df), "cohort/cache mismatch — re-run train_l1_fzark"
    te = (df["split"] == "test").values
    df_te = df[te].reset_index(drop=True)
    base = 1 / (1 + np.exp(-logits[te]))
    Y = np.stack(df["y"].values)[te]
    n = int(te.sum())

    proj = ScopeProjection(); proj.load_state_dict(torch.load("res/scope_overlay/fzarkSL.pth", map_location="cpu")); proj.eval()
    dc = load_device_config(DEVCFG)
    l1_heads = set(dc.l1_heads())
    base_heads = frozenset(dc.base_heads())            # uncalibrated heads on this device
    fire = {h: 0.5 for h in SCOPE_HEADS}; fire.update(dc.fire_thresholds())
    fire_disable = dict(fire);
    for h in base_heads:
        fire_disable[h] = NEVER                          # (b) disable base-routed heads
    with torch.no_grad():
        l1p = proj(torch.tensor(logits[te], dtype=torch.float32)).numpy()

    routed = [{h: (float(l1p[r, j]) if h in l1_heads else float(base[r, h]))
               for j, h in enumerate(SCOPE_HEADS)} | {1: float(base[r, 1])}
              for r in range(n)]

    variants = {
        "base":    np.array([[base[r, h] >= 0.5 for h in SCOPE_HEADS] for r in range(n)]),
        "current": np.array([list(decide(routed[r], fire, frozenset()).values()) for r in range(n)]),
        "exclude": np.array([list(decide(routed[r], fire, base_heads).values()) for r in range(n)]),
        "disable": np.array([list(decide(routed[r], fire_disable, base_heads).values()) for r in range(n)]),
    }

    cohort = df_te["cohort"].values; event = df_te["event"].values

    # ── per-head panels ──
    rows = []
    for j, h in enumerate(SCOPE_HEADS):
        evs = [e for e, hh in SCOPE_EVENT_TO_HEAD.items() if hh == h]
        tp_m = (Y[:, j] == 1)
        fp_m = (cohort == "FP") & np.isin(event, evs)
        rows.append((h, dc.heads[h]["source"],
                     {k: panel(v[:, j], tp_m, fp_m) for k, v in variants.items()}))

    # ── arbitration cross-talk diagnostics (current vs exclude) ──
    gidx = [SCOPE_HEADS.index(x) for x in RATE_GROUP]
    cur, exc = variants["current"], variants["exclude"]
    changed = (cur != exc).any(axis=1)
    # records where uncalibrated head co-fired (pre-arbitration) with a calibrated peer
    pre = np.array([[routed[r][x] >= fire[x] for x in RATE_GROUP] for r in range(n)])
    uncal_in_group = [x for x in RATE_GROUP if x in base_heads]
    cofire = np.zeros(n, bool)
    if uncal_in_group:
        u = RATE_GROUP.index(uncal_in_group[0])
        cal = [i for i in range(len(RATE_GROUP)) if i != u]
        cofire = pre[:, u] & pre[:, cal].any(axis=1)

    # ── report ──
    md = ["# fzark TP/FP — base-routed-head L2 refinement evaluation",
          "\n**Recommendation under test:** on a device where a head is base-routed "
          "(uncalibrated, base@0.5), exclude it from L2 rate-exclusion arbitration "
          "(`exclude`) — or disable it outright (`disable`). On the fzark config the only "
          f"base-routed scope head is **{', '.join(L.DETECTION_SCOPE[h] for h in base_heads)}** "
          f"(head {','.join(map(str, base_heads))}). It shares the rate-exclusion group "
          "{Brady 4, AFib 5, Tachy 6}, so the change can move decisions for the uncalibrated "
          "head AND its calibrated peers.\n",
          "Held-out fzark TEST split (seed-fixed, grouped by Source Data File). "
          "TP→sensitivity, FP→retention/specificity, pooled PPV/F1 **relative** "
          "(cohorts separately capped-sampled). All prod variants use the fzark device "
          "config + L2 GT-matched (ON).\n",
          f"Test records: {n}.  Rate-exclusion-group co-fires where the uncalibrated head "
          f"({','.join(map(str, uncal_in_group)) or 'none'}) fired alongside a calibrated "
          f"peer pre-arbitration: **{int(cofire.sum())}**.  Records whose final decisions "
          f"differ (current→exclude): **{int(changed.sum())}**.\n"]

    md.append("| head | source | TP n | FP n | metric | base | current | **exclude (a)** | disable (b) |")
    md.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for h, src, P in rows:
        nm = L.DETECTION_SCOPE[h]; first = True
        for met, lab in (("sens", "Sensitivity"), ("spec", "Specificity"),
                         ("fp_ret", "FP-retention"), ("ppv", "PPV*"), ("f1", "F1*")):
            def fmt(k): return "—" if P[k][met] != P[k][met] else f"{P[k][met]:.3f}"
            hc = f"{h} {nm}" if first else ""
            sc = (src + (" ★" if h in base_heads else "")) if first else ""
            tpn = P["base"]["npos"] if first else ""; fpn = P["base"]["nneg"] if first else ""
            md.append(f"| {hc} | {sc} | {tpn} | {fpn} | {lab} | {fmt('base')} | {fmt('current')} | "
                      f"**{fmt('exclude')}** | {fmt('disable')} |")
            first = False

    md.append("\n★ = base-routed (uncalibrated) head on this device.\n")
    md.append("## Interpretation\n")
    md.append("- **`current`** lets the uncalibrated head suppress / be suppressed by calibrated "
              "peers — those decisions are effectively arbitrary, since base@0.5 on OOD "
              "single-lead is not a meaningful operating point.")
    md.append("- **`exclude (a)`** removes the uncalibrated head from rate-exclusion only; it "
              "still fires on its own threshold. Calibrated peers (Brady/AFib) are no longer "
              "perturbed by it. This is the implemented device-config refinement.")
    md.append("- **`disable (b)`** drops the uncalibrated head's alerts entirely. On fzark the "
              "head has 0 TP events, so disabling costs **zero** sensitivity while removing its "
              "false alarms — the strongest specificity option when a head is uncalibrated.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
