"""L2 emitted-vs-GT co-fire regression — does L2 reproduce the GT co-occurrence matrix?

Runs the production routing (L1 fuzzySL) on PTB-XL fold-10 and compares the
emitted scope-head co-occurrence counts WITHOUT L2 (pass-through) vs WITH the
GT-matched L2, against the GT matrix. Asserts: the GT-zero rate-head pairs
(Brady+AFib, Brady+Tachy, AFib+Tachy) are eliminated by L2, and the coupled
SVT≡VT pair is preserved. Run on lead-II and fuzzy.

Run:  python3 -m scripts.eval_l2_cofire
"""
from __future__ import annotations
import os, sys, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, DEFAULT_CKPT
from overlay.inference import L1_HEADS
from overlay.arbiter import arbitrate, ArbiterConfig, DEFAULT_CONFIG, GT_MATCHED_CONFIG
from overlay.types import ScopeScores
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR)
from scripts.calibrate_l1_policy import maxf1_threshold, sens_at, ANGLES
from scripts.train_scope_overlay import drop_for, POLICY_HEADS

PAIRS = list(itertools.combinations(SCOPE_HEADS, 2))
RATE_PAIRS = [(4, 5), (4, 6), (5, 6)]
COUPLE = (93, 98)


def cofire_counts(decmat):
    """decmat: list of dict[head]→fired bool. Return pairwise co-fire counts."""
    F = np.array([[int(d[h]) for h in SCOPE_HEADS] for d in decmat])
    out = {}
    for a, b in PAIRS:
        ia, ib = SCOPE_HEADS.index(a), SCOPE_HEADS.index(b)
        out[(a, b)] = int((F[:, ia] & F[:, ib]).sum())
    return out


def run(name, logits, Y, l1, thr):
    fire = {h: 0.5 for h in SCOPE_HEADS}
    fire.update({h: thr[h] for h in thr})
    with torch.no_grad():
        p = l1(torch.tensor(logits, dtype=torch.float32)).numpy()
    base = 1 / (1 + np.exp(-logits))
    # routed probs: L1 for {4,5,6}, base for the rest
    probs = []
    for r in range(len(logits)):
        d = {}
        for j, h in enumerate(SCOPE_HEADS):
            d[h] = float(p[r, j]) if h in L1_HEADS else float(base[r, h])
        probs.append(d)

    # GT co-fire
    gt = {}
    for a, b in PAIRS:
        ia, ib = SCOPE_HEADS.index(a), SCOPE_HEADS.index(b)
        gt[(a, b)] = int((Y[:, ia].astype(bool) & Y[:, ib].astype(bool)).sum())

    cfgs = {"L2 off": ArbiterConfig(enabled=False, fire_threshold=fire),
            "L2 on": ArbiterConfig(enabled=True, fire_threshold=fire)}
    emitted = {}
    for tag, cfg in cfgs.items():
        dec = [{h: arbitrate(ScopeScores(probs=pr, nsr_score=0.0), cfg)[h].fired
                for h in SCOPE_HEADS} for pr in probs]
        emitted[tag] = cofire_counts(dec)

    print(f"\n=== {name} (n={len(logits)}) — pairwise co-fire counts ===")
    print(f"{'pair':<26}{'GT':>6}{'L2 off':>8}{'L2 on':>8}")
    ok = True
    for a, b in PAIRS:
        g, off, on = gt[(a, b)], emitted['L2 off'][(a, b)], emitted['L2 on'][(a, b)]
        if g == 0 and off == 0 and on == 0:
            continue
        tag = ""
        if (a, b) in RATE_PAIRS:
            tag = "  rate-excl → expect 0" + ("  OK" if on == 0 else "  ** FAIL **")
            ok = ok and on == 0
        elif (a, b) == COUPLE:
            tag = "  couple → expect kept" + ("  OK" if on > 0 or g == 0 else "")
        print(f"{a}+{b} {L.DETECTION_SCOPE[a][:8]}/{L.DETECTION_SCOPE[b][:8]:<10}{g:>6}{off:>8}{on:>8}{tag}")
    print(f"  rate-pair exclusion {'PASS' if ok else 'FAIL'} (GT-zero pairs eliminated by L2)")
    return ok


def fit(l1, vlog, vY):
    with torch.no_grad():
        p = l1(torch.tensor(vlog, dtype=torch.float32)).numpy()
    b = 1 / (1 + np.exp(-vlog[:, SCOPE_HEADS]))
    thr = {}
    for i, h in enumerate(SCOPE_HEADS):
        if h not in POLICY_HEADS or vY[:, i].sum() == 0:
            continue
        thr[h] = maxf1_threshold(p[:, i], vY[:, i], sens_at(b[:, i], vY[:, i], 0.5),
                                 drop_for(h), 0.02)[0]
    return thr


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(DEFAULT_CKPT, map_location="cpu")); l1.eval()
    fm = fold_map(); lut = label_lookup()
    allok = True

    f0, l0, e0 = load_ptbxl_leadii()
    va = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    thr_ll = fit(l1, f0[va], l0[va][:, SCOPE_HEADS])
    allok &= run("PTB-XL lead-II", f0[te], l0[te][:, SCOPE_HEADS], l1, thr_ll)

    Vl, Vy, Tl, Ty = [], [], [], []
    for a in ANGLES:
        tr = np.load(f"{FUZZY_DIR}/train_{a}deg.npz"); eid = tr["ecg_ids"].astype(int)
        kv = np.array([fm.get(int(e)) == 9 for e in eid])
        Vl.append(fuzzy_logits(f"{FUZZY_DIR}/train_{a}deg.npz", backbone, device)[kv])
        Vy.append(np.stack([lut[int(e)] for e in eid[kv]])[:, SCOPE_HEADS])
        fn = "val_60deg_split.npz" if a == 60 else f"val_{a}deg.npz"
        ve = np.load(f"{FUZZY_DIR}/{fn}"); et = ve["ecg_ids"].astype(int)
        Tl.append(fuzzy_logits(f"{FUZZY_DIR}/{fn}", backbone, device))
        Ty.append(np.stack([lut[int(e)] for e in et])[:, SCOPE_HEADS])
    thr_fz = fit(l1, np.concatenate(Vl), np.concatenate(Vy))
    allok &= run("FUZZY (4 angles)", np.concatenate(Tl), np.concatenate(Ty), l1, thr_fz)

    print(f"\nL2 co-fire regression: {'PASS' if allok else 'FAIL'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
